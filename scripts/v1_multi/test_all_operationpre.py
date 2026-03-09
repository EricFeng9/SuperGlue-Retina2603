"""
针对 train_onMultiGen_vessels_enhanced.py 训练权重的测试脚本

测试三个数据集的全量数据（train+val 合并）：
  - operation_pre_filtered_cffa  (CFFA)
  - operation_pre_filtered_cfoct (CFOCT)
  - operation_pre_filtered_octfa (OCTFA)

支持 --baseline 参数，同时用 SuperGlue 原生预训练权重跑一遍，
最终输出 trained vs baseline 各项指标的对比表格 (CSV)。
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from loguru import logger
import argparse
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.superglue import SuperGlue
from models.superpoint import SuperPoint

from scripts.v1_multi.metrics import (
    compute_homography_errors,
    set_metrics_verbose,
    error_auc,
    compute_auc_rop
)

from data.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from data.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from data.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset


# ---------------------------------------------------------------------------
# 单应矩阵合法性检查
# ---------------------------------------------------------------------------

def is_valid_homography(H, scale_min=0.1, scale_max=10.0, perspective_threshold=0.005):
    if H is None:
        return False
    if np.isnan(H).any() or np.isinf(H).any():
        return False
    det = np.linalg.det(H[:2, :2])
    if det < scale_min or det > scale_max:
        return False
    if abs(H[2, 0]) > perspective_threshold or abs(H[2, 1]) > perspective_threshold:
        return False
    return True


# ---------------------------------------------------------------------------
# 数据集 Wrapper：把真实数据集转换为模型输入格式
# ---------------------------------------------------------------------------

class RealDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, split_name='unknown', dataset_name='MultiModal'):
        self.base_dataset = base_dataset
        self.split_name = split_name
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        moving_original_tensor = (moving_original_tensor + 1) / 2
        moving_gt_tensor = (moving_gt_tensor + 1) / 2

        def to_gray(t):
            if t.shape[0] == 3:
                return (0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]).unsqueeze(0)
            return t

        fix_gray = to_gray(fix_tensor)
        moving_gray = to_gray(moving_gt_tensor)
        moving_orig_gray = to_gray(moving_original_tensor)

        fix_name = os.path.basename(fix_path)
        moving_name = os.path.basename(moving_path)

        try:
            T_fix_to_moving = torch.inverse(T_0to1)
        except Exception:
            T_fix_to_moving = T_0to1

        return {
            'image0': fix_gray,
            'image1': moving_orig_gray,
            'image1_gt': moving_gray,
            'T_0to1': T_fix_to_moving,
            'pair_names': (fix_name, moving_name),
            'dataset_name': self.dataset_name,
            'split': self.split_name
        }


# ---------------------------------------------------------------------------
# 评估器
# ---------------------------------------------------------------------------

def compute_metrics_for_dataset(evaluator, dataset_name):
    """AUC 使用全部样本（含 Failed/Inaccurate）；MSE/MACE 仅对 Acceptable 样本求平均。
    若某模态上 Trained 的 Inaccurate 更多，则 AUC 可能低于 Baseline，而 MSE/MACE 仍可更优。"""
    total = evaluator.per_dataset_samples.get(dataset_name, 0)
    if total == 0:
        return None

    errors = evaluator.per_dataset_errors.get(dataset_name, [])
    mses = evaluator.per_dataset_mses.get(dataset_name, [])
    maces = evaluator.per_dataset_maces.get(dataset_name, [])
    failed = evaluator.per_dataset_failed.get(dataset_name, 0)
    inaccurate = evaluator.per_dataset_inaccurate.get(dataset_name, 0)
    acceptable = evaluator.per_dataset_acceptable.get(dataset_name, 0)

    if len(errors) > 0:
        auc_dict = error_auc(errors, [5, 10, 20])
        mauc_dict = compute_auc_rop(errors, limit=25)
        combined = (auc_dict.get('auc@5', 0.0) + auc_dict.get('auc@10', 0.0) + auc_dict.get('auc@20', 0.0)) / 3.0
    else:
        auc_dict = {}
        mauc_dict = {}
        combined = 0.0

    return {
        'dataset': dataset_name,
        'auc@5': auc_dict.get('auc@5', 0.0),
        'auc@10': auc_dict.get('auc@10', 0.0),
        'auc@20': auc_dict.get('auc@20', 0.0),
        'mAUC': mauc_dict.get('mAUC', 0.0),
        'combined_auc': combined,
        'mse': sum(mses) / len(mses) if mses else 0.0,
        'mace': sum(maces) / len(maces) if maces else 0.0,
        'num_samples': total,
        'failed': failed,
        'inaccurate': inaccurate,
        'acceptable': acceptable,
    }


class UnifiedEvaluator:
    def __init__(self, config=None):
        self.config = config
        self.reset()

    def reset(self):
        self.all_errors = []
        self.all_mses = []
        self.all_maces = []
        self.total_samples = 0
        self.failed_samples = 0
        self.inaccurate_samples = 0
        self.acceptable_samples = 0

        self.per_dataset_errors = {}
        self.per_dataset_mses = {}
        self.per_dataset_maces = {}
        self.per_dataset_samples = {}
        self.per_dataset_failed = {}
        self.per_dataset_inaccurate = {}
        self.per_dataset_acceptable = {}

    def evaluate_batch(self, batch, outputs, pl_module):
        matches0 = outputs['matches0']
        kpts0 = batch['keypoints0']
        kpts1 = batch['keypoints1']
        dataset_names = batch.get('dataset_name', ['unknown'] * kpts0.shape[0])

        B = kpts0.shape[0]

        mkpts0_f_list, mkpts1_f_list, m_bids_list = [], [], []

        for b in range(B):
            m0 = matches0[b]
            valid = m0 > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[valid]

            pts0 = kpts0[b][m_indices_0].cpu().numpy()
            pts1 = kpts1[b][m_indices_1].cpu().numpy()

            if len(pts0) > 0:
                mkpts0_f_list.append(torch.from_numpy(pts0).float())
                mkpts1_f_list.append(torch.from_numpy(pts1).float())
                m_bids_list.append(torch.full((len(pts0),), b, dtype=torch.long))

        metrics_batch = {
            'mkpts0_f': torch.cat(mkpts0_f_list, dim=0) if mkpts0_f_list else torch.empty(0, 2),
            'mkpts1_f': torch.cat(mkpts1_f_list, dim=0) if mkpts1_f_list else torch.empty(0, 2),
            'm_bids': torch.cat(m_bids_list, dim=0) if m_bids_list else torch.empty(0, dtype=torch.long),
            'T_0to1': batch['T_0to1'],
            'image0': batch['image0'],
            'dataset_name': batch['dataset_name']
        }

        compute_homography_errors(metrics_batch, self.config if self.config else pl_module.config)

        self.total_samples += B
        failed_mask = metrics_batch.get('failed_mask', [False] * B)
        inaccurate_mask = metrics_batch.get('inaccurate_mask', [False] * B)

        self.failed_samples += int(np.sum(np.array(failed_mask, dtype=np.int64)))
        self.inaccurate_samples += int(np.sum(np.array(inaccurate_mask, dtype=np.int64)))
        self.acceptable_samples += int(B - np.sum(np.array(failed_mask, dtype=np.int64)) - np.sum(np.array(inaccurate_mask, dtype=np.int64)))

        if len(metrics_batch.get('t_errs', [])) > 0:
            self.all_errors.extend(list(metrics_batch['t_errs']))

        batch_mses = list(metrics_batch.get('mse_list', []))
        batch_maces = list(metrics_batch.get('mace_list', []))
        for mse in batch_mses:
            if np.isfinite(mse):
                self.all_mses.append(float(mse))
        for mace in batch_maces:
            if np.isfinite(mace):
                self.all_maces.append(float(mace))

        for b in range(B):
            dataset = dataset_names[b] if isinstance(dataset_names, list) else dataset_names
            if dataset not in self.per_dataset_errors:
                self.per_dataset_errors[dataset] = []
                self.per_dataset_mses[dataset] = []
                self.per_dataset_maces[dataset] = []
                self.per_dataset_samples[dataset] = 0
                self.per_dataset_failed[dataset] = 0
                self.per_dataset_inaccurate[dataset] = 0
                self.per_dataset_acceptable[dataset] = 0

            self.per_dataset_samples[dataset] += 1
            if b < len(failed_mask):
                if failed_mask[b]:
                    self.per_dataset_failed[dataset] += 1
                elif inaccurate_mask[b]:
                    self.per_dataset_inaccurate[dataset] += 1
                else:
                    self.per_dataset_acceptable[dataset] += 1

            # 与 MACE/MSE 一致：仅当样本未 failed 且未 inaccurate 时才计入 AUC，避免“训练后 AUC 变差、MACE 变好”的假象
            # AUC 包含所有样本：Failed = 1e6, Inaccurate = 真实误差, Success = 真实误差（与 metrics_cau_principle_0305.md 一致）
            if b < len(metrics_batch.get('t_errs', [])):
                self.per_dataset_errors[dataset].append(metrics_batch['t_errs'][b])

            if b < len(batch_mses) and np.isfinite(batch_mses[b]):
                self.per_dataset_mses[dataset].append(float(batch_mses[b]))
            if b < len(batch_maces) and np.isfinite(batch_maces[b]):
                self.per_dataset_maces[dataset].append(float(batch_maces[b]))

        return {
            'H_est': metrics_batch.get('H_est', [np.eye(3)] * B),
            'mses': batch_mses,
            'maces': batch_maces,
            'metrics_batch': metrics_batch,
            'matches0': matches0,
            'kpts0': kpts0,
            'kpts1': kpts1
        }

    def compute_epoch_metrics(self):
        metrics = {}

        if self.all_errors and len(self.all_errors) > 0:
            auc_dict = error_auc(self.all_errors, [5, 10, 20])
            metrics['auc@5'] = auc_dict.get('auc@5', 0.0)
            metrics['auc@10'] = auc_dict.get('auc@10', 0.0)
            metrics['auc@20'] = auc_dict.get('auc@20', 0.0)
            mauc_dict = compute_auc_rop(self.all_errors, limit=25)
            metrics['mAUC'] = mauc_dict.get('mAUC', 0.0)
        else:
            metrics['auc@5'] = 0.0
            metrics['auc@10'] = 0.0
            metrics['auc@20'] = 0.0
            metrics['mAUC'] = 0.0

        metrics['combined_auc'] = (metrics['auc@5'] + metrics['auc@10'] + metrics['auc@20']) / 3.0
        metrics['mse'] = sum(self.all_mses) / len(self.all_mses) if self.all_mses else 0.0
        metrics['mace'] = sum(self.all_maces) / len(self.all_maces) if self.all_maces else 0.0
        metrics['inverse_mace'] = 1.0 / (1.0 + metrics['mace']) if metrics['mace'] > 0 else 1.0
        metrics['match_failure_rate'] = self.failed_samples / self.total_samples if self.total_samples > 0 else 0.0
        metrics['total_samples'] = self.total_samples
        metrics['failed_samples'] = self.failed_samples
        metrics['success_samples'] = self.total_samples - self.failed_samples
        metrics['inaccurate_samples'] = self.inaccurate_samples
        metrics['acceptable_samples'] = self.acceptable_samples
        metrics['inaccurate_rate'] = self.inaccurate_samples / self.total_samples if self.total_samples > 0 else 0.0
        metrics['acceptable_rate'] = self.acceptable_samples / self.total_samples if self.total_samples > 0 else 0.0

        metrics['per_dataset'] = {}
        for dataset_name in self.per_dataset_errors.keys():
            ds_metrics = compute_metrics_for_dataset(self, dataset_name)
            if ds_metrics:
                metrics['per_dataset'][dataset_name] = ds_metrics

        return metrics


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def create_chessboard(img1, img2, grid_size=4):
    H, W = img1.shape
    cell_h = H // grid_size
    cell_w = W // grid_size
    chessboard = np.zeros((H, W), dtype=img1.dtype)
    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w
            if (i + j) % 2 == 0:
                chessboard[y_start:y_end, x_start:x_end] = img1[y_start:y_end, x_start:x_end]
            else:
                chessboard[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
    return chessboard


def _visualize_sample(batch, outputs, output_dir, batch_idx, sample_idx):
    """可视化单个样本"""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    H_ests = outputs.get('H_est', [np.eye(3)])
    dataset_names = batch.get('dataset_name', ['unknown'])

    H_est = H_ests[sample_idx] if sample_idx < len(H_ests) else np.eye(3)
    if not is_valid_homography(H_est):
        H_est = np.eye(3)

    img0 = (batch['image0'][sample_idx, 0].cpu().numpy() * 255).astype(np.uint8)
    img1 = (batch['image1'][sample_idx, 0].cpu().numpy() * 255).astype(np.uint8)
    img1_gt = (batch['image1_gt'][sample_idx, 0].cpu().numpy() * 255).astype(np.uint8)

    h, w = img0.shape
    try:
        H_inv = np.linalg.inv(H_est)
        img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
    except Exception:
        img1_result = img1.copy()

    dataset_name = dataset_names[sample_idx] if isinstance(dataset_names, list) else dataset_names

    pair_names = batch.get('pair_names', None)
    if pair_names:
        sample_name = f"{dataset_name}_batch{batch_idx:04d}_sample{sample_idx:02d}_{Path(pair_names[0][sample_idx]).stem}_vs_{Path(pair_names[1][sample_idx]).stem}"
    else:
        sample_name = f"{dataset_name}_batch{batch_idx:04d}_sample{sample_idx:02d}"

    save_path = output_dir / sample_name
    save_path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(save_path / "fix.png"), img0)
    cv2.imwrite(str(save_path / "moving_original.png"), img1)
    cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
    cv2.imwrite(str(save_path / "moving_gt.png"), img1_gt)

    img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    if 'kpts0' in outputs and 'kpts1' in outputs:
        kpts0_np = outputs['kpts0'][sample_idx].cpu().numpy()
        kpts1_np = outputs['kpts1'][sample_idx].cpu().numpy()

        for pt in kpts0_np:
            cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
        for pt in kpts1_np:
            cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)

        if 'matches0' in outputs:
            m0 = outputs['matches0'][sample_idx].cpu()
            valid = m0 > -1
            m_indices_0 = torch.where(valid)[0].numpy()
            m_indices_1 = m0[valid].numpy()

            for idx0 in m_indices_0:
                pt = kpts0_np[idx0]
                cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
            for idx1 in m_indices_1:
                pt = kpts1_np[idx1]
                cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

            try:
                margin = 10
                h1, w1 = img0.shape[:2]
                h2, w2 = img1.shape[:2]
                H_canvas = max(h1, h2)
                W_canvas = w1 + w2 + margin

                out_img = np.zeros((H_canvas, W_canvas), dtype=np.uint8)
                out_img[:h1, :w1] = img0
                out_img[:h2, w1 + margin:w1 + margin + w2] = img1

                fig = plt.figure(figsize=(12, 6))
                plt.imshow(out_img, cmap='gray')
                plt.axis('off')

                if len(m_indices_0) > 0:
                    pts0 = kpts0_np[m_indices_0]
                    pts1 = kpts1_np[m_indices_1] + np.array([w1 + margin, 0])
                    for p0, p1 in zip(pts0, pts1):
                        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], color='lime', lw=0.5)

                plt.savefig(str(save_path / "matches.png"), bbox_inches='tight', dpi=100)
                plt.close(fig)
            except Exception as e:
                logger.warning(f"绘制匹配图失败: {e}")

    cv2.imwrite(str(save_path / "fix_with_kpts.png"), img0_color)
    cv2.imwrite(str(save_path / "moving_with_kpts.png"), img1_color)

    try:
        cb = create_chessboard(img1_result, img0)
        cv2.imwrite(str(save_path / "chessboard.png"), cb)
        cb_gt_vs_fix = create_chessboard(img1_gt, img0)
        cv2.imwrite(str(save_path / "chessboard_gt_vs_fix.png"), cb_gt_vs_fix)
        cb_gt_vs_pred = create_chessboard(img1_gt, img1_result)
        cv2.imwrite(str(save_path / "chessboard_gt_vs_pred.png"), cb_gt_vs_pred)
    except Exception:
        pass

    if 'mses' in outputs and sample_idx < len(outputs['mses']):
        mse_val = outputs['mses'][sample_idx]
        mace_val = outputs['maces'][sample_idx]
        with open(save_path / "metrics.txt", "w") as f:
            f.write(f"MSE: {mse_val:.6f}\n")
            f.write(f"MACE: {mace_val:.4f}\n")
            if 'matches0' in outputs:
                m0 = outputs['matches0'][sample_idx].cpu()
                num_matches = int(torch.sum(m0 > -1).item())
                f.write(f"Matches: {num_matches}\n")


# ---------------------------------------------------------------------------
# 评估主循环（带每数据集最多 N 个可视化）
# ---------------------------------------------------------------------------

def run_evaluation(pl_module, dataloader, config, label='model',
                   save_visualizations=False, output_dir=None, max_viz_per_dataset=5):
    """
    运行完整的评估流程，可视化最多 max_viz_per_dataset 个样本/每个数据集。
    """
    evaluator = UnifiedEvaluator(config=config)
    viz_counts = {}   # {dataset_name: int}

    pl_module.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = pl_module(batch)
            result = evaluator.evaluate_batch(batch, outputs, pl_module)

            if save_visualizations and output_dir:
                batch_size = batch['image0'].shape[0]
                dataset_names = batch.get('dataset_name', ['unknown'] * batch_size)
                for s_idx in range(batch_size):
                    ds = dataset_names[s_idx] if isinstance(dataset_names, list) else dataset_names
                    if viz_counts.get(ds, 0) < max_viz_per_dataset:
                        _visualize_sample(batch, result, output_dir, batch_idx, s_idx)
                        viz_counts[ds] = viz_counts.get(ds, 0) + 1

            if batch_idx % 10 == 0:
                logger.info(f"[{label}] 已处理 {batch_idx + 1} 个 batch")

    metrics = evaluator.compute_epoch_metrics()
    logger.info(f"[{label}] 评估完成")
    return metrics


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    'CFFA': {
        'dir': 'operation_pre_filtered_cffa',
        'cls': CFFADataset,
        'kwargs_train': {'mode': 'cf2fa', 'split': 'train'},
        'kwargs_val': {'mode': 'cf2fa', 'split': 'val'},
    },
    'CFOCT': {
        'dir': 'operation_pre_filtered_cfoct',
        'cls': CFOCTDataset,
        'kwargs_train': {'mode': 'cf2oct', 'split': 'train'},
        'kwargs_val': {'mode': 'cf2oct', 'split': 'val'},
    },
    'OCTFA': {
        'dir': 'operation_pre_filtered_octfa',
        'cls': OCTFADataset,
        'kwargs_train': {'mode': 'fa2oct', 'split': 'train'},
        'kwargs_val': {'mode': 'fa2oct', 'split': 'val'},
    },
}


def build_full_dataloader(dataset_name, batch_size, num_workers):
    """为指定数据集构建全量（train+val）DataLoader"""
    script_dir = Path(__file__).parent.parent.parent
    cfg = DATASET_CONFIGS[dataset_name]
    data_dir = str(script_dir / 'data' / cfg['dir'])

    train_base = cfg['cls'](root_dir=data_dir, **cfg['kwargs_train'])
    val_base = cfg['cls'](root_dir=data_dir, **cfg['kwargs_val'])

    train_wrap = RealDatasetWrapper(train_base, split_name='train', dataset_name=dataset_name)
    val_wrap = RealDatasetWrapper(val_base, split_name='val', dataset_name=dataset_name)

    full_dataset = ConcatDataset([train_wrap, val_wrap])
    logger.info(f"数据集 {dataset_name}: train={len(train_wrap)}, val={len(val_wrap)}, total={len(full_dataset)}")

    return DataLoader(full_dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# Baseline 模型
# ---------------------------------------------------------------------------

def build_baseline_model(config):
    """构建使用 SuperGlue 原生预训练权重的 baseline 模型"""

    class BaselineSuperGlueModel(pl.LightningModule):
        def __init__(self, config):
            super().__init__()
            self.config = config

            sp_config = config.SUPERPOINT.copy()
            pretrained_path = config.SUPERPOINT_PRETRAINED
            self.extractor = SuperPoint(sp_config, pretrained_path=pretrained_path).eval()
            for param in self.extractor.parameters():
                param.requires_grad = False

            sg_conf = config.SUPERGLUE.copy()
            sg_conf['weights'] = 'indoor'
            self.matcher = SuperGlue(sg_conf).eval()
            for param in self.matcher.parameters():
                param.requires_grad = False

        def forward(self, batch):
            with torch.no_grad():
                if 'keypoints0' not in batch:
                    feats0 = self.extractor({'image': batch['image0']})
                    feats1 = self.extractor({'image': batch['image1']})

                    def list_to_batch(feats):
                        keys = feats['keypoints']
                        descs = feats['descriptors']
                        scs = feats['scores']

                        lengths = [int(k.shape[0]) for k in keys]
                        n_common = min(lengths) if lengths else 0

                        if n_common <= 0:
                            device = keys[0].device if len(keys) > 0 else batch['image0'].device
                            return (
                                torch.zeros((len(lengths), 0, 2), device=device, dtype=torch.float32),
                                torch.zeros((len(lengths), 256, 0), device=device, dtype=torch.float32),
                                torch.zeros((len(lengths), 0), device=device, dtype=torch.float32),
                            )

                        kpts_b, desc_b, scores_b = [], [], []
                        for k, d, s in zip(keys, descs, scs):
                            k, d, s = k.float(), d.float(), s.float()
                            if k.shape[0] > n_common:
                                topk = torch.topk(s, k=n_common, dim=0, largest=True, sorted=False).indices
                                k, s, d = k[topk], s[topk], d[:, topk]
                            kpts_b.append(k)
                            desc_b.append(d)
                            scores_b.append(s)

                        return (
                            torch.stack(kpts_b, dim=0),
                            torch.stack(desc_b, dim=0),
                            torch.stack(scores_b, dim=0),
                        )

                    kpts0, desc0, sc0 = list_to_batch(feats0)
                    kpts1, desc1, sc1 = list_to_batch(feats1)
                    batch.update({
                        'keypoints0': kpts0, 'descriptors0': desc0, 'scores0': sc0,
                        'keypoints1': kpts1, 'descriptors1': desc1, 'scores1': sc1,
                    })

            data = {
                'descriptors0': batch['descriptors0'],
                'descriptors1': batch['descriptors1'],
                'keypoints0': batch['keypoints0'],
                'keypoints1': batch['keypoints1'],
                'scores0': batch['scores0'],
                'scores1': batch['scores1'],
                'image0': batch['image0'],
                'image1': batch['image1'],
            }
            return self.matcher(data)

    model = BaselineSuperGlueModel(config)
    model.eval()
    logger.info("已加载 SuperGlue 原生预训练权重 (indoor) 作为 Baseline")
    return model


# ---------------------------------------------------------------------------
# 结果保存
# ---------------------------------------------------------------------------

DATASET_ORDER = ['CFFA', 'CFOCT', 'OCTFA']
METRIC_COLS = ['num_samples', 'failed', 'inaccurate', 'acceptable', 'auc@5', 'auc@10', 'auc@20', 'mAUC', 'combined_auc', 'mse', 'mace']
METRIC_FMT = {
    'num_samples': lambda v: str(int(v)),
    'failed': lambda v: str(int(v)),
    'inaccurate': lambda v: str(int(v)),
    'acceptable': lambda v: str(int(v)),
    'auc@5': lambda v: f"{v:.4f}",
    'auc@10': lambda v: f"{v:.4f}",
    'auc@20': lambda v: f"{v:.4f}",
    'mAUC': lambda v: f"{v:.4f}",
    'combined_auc': lambda v: f"{v:.4f}",
    'mse': lambda v: f"{v:.6f}",
    'mace': lambda v: f"{v:.4f}",
}


def save_summary_txt(output_dir, label, ds_metrics_map):
    summary_path = output_dir / f"test_summary_{label}.txt"
    with open(summary_path, "w") as f:
        f.write(f"测试总结 [{label}]\n")
        f.write("=" * 60 + "\n")
        f.write("说明: AUC 使用全部样本(Failed=1e6+Inaccurate+Acceptable)；MSE/MACE 仅对 Acceptable 求平均。\n")
        f.write("若某模态 Trained 的 Inaccurate 多于 Baseline，则可能出现 AUC 更低但 MSE/MACE 更优。\n")
        f.write("-" * 60 + "\n")
        for ds in DATASET_ORDER:
            if ds not in ds_metrics_map:
                continue
            m = ds_metrics_map[ds]
            f.write(f"\n[{ds}] 样本数: {m['num_samples']} (Failed: {m.get('failed', 0)}, Inaccurate: {m.get('inaccurate', 0)}, Acceptable: {m.get('acceptable', 0)})\n")
            f.write(f"  AUC@5:        {m['auc@5']:.4f}\n")
            f.write(f"  AUC@10:       {m['auc@10']:.4f}\n")
            f.write(f"  AUC@20:       {m['auc@20']:.4f}\n")
            f.write(f"  mAUC:         {m['mAUC']:.4f}\n")
            f.write(f"  Combined AUC: {m['combined_auc']:.4f}\n")
            f.write(f"  MSE:          {m['mse']:.6f}\n")
            f.write(f"  MACE:         {m['mace']:.4f}\n")
    logger.info(f"测试总结已保存: {summary_path}")


def save_comparison_csv(output_dir, trained_metrics, baseline_metrics=None):
    """
    保存对比 CSV，列为: Dataset, Metric, Trained, Baseline (可选)
    """
    csv_path = output_dir / "comparison.csv"
    header = ['Dataset', 'Metric', 'Trained']
    if baseline_metrics is not None:
        header.append('Baseline')

    rows = []
    for ds in DATASET_ORDER:
        t_ds = trained_metrics.get(ds, {})
        b_ds = baseline_metrics.get(ds, {}) if baseline_metrics else {}
        for metric in METRIC_COLS:
            t_val = METRIC_FMT[metric](t_ds.get(metric, 0.0)) if t_ds else 'N/A'
            row = [ds, metric, t_val]
            if baseline_metrics is not None:
                b_val = METRIC_FMT[metric](b_ds.get(metric, 0.0)) if b_ds else 'N/A'
                row.append(b_val)
            rows.append(row)

    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info(f"对比 CSV 已保存: {csv_path}")

    # 打印对比表格到日志
    _print_comparison_table(trained_metrics, baseline_metrics)

    return csv_path


def _print_comparison_table(trained_metrics, baseline_metrics=None):
    logger.info("说明: AUC 含全部样本；MSE/MACE 仅 Acceptable。某模态若 Trained 的 Inaccurate 更多，则可能 AUC 更低但 MSE/MACE 更优。")
    header = f"{'Dataset':<8} {'Metric':<14} {'Trained':>12}"
    if baseline_metrics is not None:
        header += f" {'Baseline':>12}"
    logger.info("=" * len(header))
    logger.info("对比表格")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))

    for ds in DATASET_ORDER:
        t_ds = trained_metrics.get(ds, {})
        b_ds = baseline_metrics.get(ds, {}) if baseline_metrics else {}
        for metric in METRIC_COLS:
            t_val = METRIC_FMT[metric](t_ds.get(metric, 0.0)) if t_ds else 'N/A'
            line = f"{ds:<8} {metric:<14} {t_val:>12}"
            if baseline_metrics is not None:
                b_val = METRIC_FMT[metric](b_ds.get(metric, 0.0)) if b_ds else 'N/A'
                line += f" {b_val:>12}"
            logger.info(line)
        logger.info("-" * len(header))


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="针对 train_onMultiGen_vessels_enhanced 权重的测试脚本，对三个全量数据集评估"
    )

    parser.add_argument('--name', '-n', type=str, required=True,
                        help='模型名称（对应 results/superglue_gen/<name>/ 目录）')
    parser.add_argument('--test_name', '-t', type=str, default='test_operationpre_full',
                        help='测试名称（结果保存子目录，默认 test_operationpre_full）')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='检查点路径（默认 results/superglue_gen/<name>/best_checkpoint/model.ckpt）')
    parser.add_argument('--baseline', action='store_true',
                        help='同时使用 SuperGlue 原生预训练权重跑 baseline，输出对比结果')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--no_viz', action='store_true', help='禁用可视化')
    parser.add_argument('--max_viz', type=int, default=5,
                        help='每个数据集保存的可视化样本数（默认 5）')
    parser.add_argument('--superpoint_pretrained', type=str, default=None,
                        help='SuperPoint 预训练权重路径（默认从 models/weights/ 加载）')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    import importlib
    module = importlib.import_module('scripts.v1_multi.train_onMultiGen_vessels_enhanced')
    pl_class = getattr(module, 'PL_SuperGlue_Gen')
    get_default_config = getattr(module, 'get_default_config')

    config = get_default_config()
    pl.seed_everything(config.TRAINER.SEED)

    # SuperPoint 预训练权重
    if args.superpoint_pretrained:
        config.SUPERPOINT_PRETRAINED = args.superpoint_pretrained
    else:
        default_sp_path = Path(__file__).parent.parent / 'models' / 'weights' / 'superpoint_v1.pth'
        if default_sp_path.exists():
            config.SUPERPOINT_PRETRAINED = str(default_sp_path)
            logger.info(f"使用默认 SuperPoint 预训练权重: {default_sp_path}")

    # 输出目录
    output_dir = Path(f"results/superglue_gen/{args.name}/{args.test_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 日志
    log_file = output_dir / "test_log.txt"
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level="INFO")
    logger.add(log_file, format=log_format, level="INFO", mode="w")
    logger.info(f"日志将保存到: {log_file}")

    # GPU 配置
    if ',' in str(args.gpus):
        gpus_list = [int(x) for x in args.gpus.split(',')]
        _n_gpus = len(gpus_list)
    else:
        try:
            gpus_list = [int(args.gpus)]
            _n_gpus = 1
        except Exception:
            gpus_list = 'auto'
            _n_gpus = 1

    config.TRAINER.WORLD_SIZE = max(_n_gpus, 1)
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size

    logger.info(f"模型名称: {args.name}")
    logger.info(f"测试名称: {args.test_name}")
    logger.info(f"输出目录: {output_dir}")

    # ---- 构建 DataLoader（每个数据集独立，全量数据） ----
    set_metrics_verbose(True)
    dataloaders = {}
    for ds_name in DATASET_ORDER:
        dataloaders[ds_name] = build_full_dataloader(ds_name, args.batch_size, args.num_workers)

    # ---- 加载训练模型 ----
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path(f"results/superglue_gen/{args.name}/best_checkpoint/model.ckpt")

    if not ckpt_path.exists():
        logger.error(f"检查点不存在: {ckpt_path}")
        return

    logger.info(f"加载训练检查点: {ckpt_path}")
    trained_model = pl_class.load_from_checkpoint(
        str(ckpt_path),
        config=config,
        result_dir=str(output_dir)
    )
    trained_model.eval()

    # ---- 设置 GPU ----
    device = torch.device(f"cuda:{gpus_list[0]}" if torch.cuda.is_available() else "cpu")
    trained_model = trained_model.to(device)

    # ---- 评估：训练模型 ----
    logger.info("=" * 60)
    logger.info("开始评估：训练模型")
    logger.info("=" * 60)

    trained_ds_metrics = {}
    for ds_name, dl in dataloaders.items():
        logger.info(f"  >> 数据集: {ds_name}")
        viz_dir = output_dir / "viz_trained" / ds_name if not args.no_viz else None
        if viz_dir:
            viz_dir.mkdir(parents=True, exist_ok=True)
        metrics = run_evaluation(
            trained_model, dl, config=config,
            label=f"trained/{ds_name}",
            save_visualizations=not args.no_viz,
            output_dir=viz_dir,
            max_viz_per_dataset=args.max_viz,
        )
        per_ds = metrics.get('per_dataset', {})
        if ds_name in per_ds:
            trained_ds_metrics[ds_name] = per_ds[ds_name]
        else:
            trained_ds_metrics[ds_name] = {k: 0.0 for k in METRIC_COLS}

    save_summary_txt(output_dir, 'trained', trained_ds_metrics)

    # ---- 评估：Baseline（可选） ----
    baseline_ds_metrics = None
    if args.baseline:
        logger.info("=" * 60)
        logger.info("开始评估：SuperGlue Baseline（原生预训练权重）")
        logger.info("=" * 60)

        baseline_model = build_baseline_model(config)
        baseline_model = baseline_model.to(device)

        baseline_ds_metrics = {}
        for ds_name in DATASET_ORDER:
            # 重新构建 DataLoader（避免 DataLoader 状态复用问题）
            dl = build_full_dataloader(ds_name, args.batch_size, args.num_workers)
            logger.info(f"  >> 数据集: {ds_name}")
            viz_dir = output_dir / "viz_baseline" / ds_name if not args.no_viz else None
            if viz_dir:
                viz_dir.mkdir(parents=True, exist_ok=True)
            metrics = run_evaluation(
                baseline_model, dl, config=config,
                label=f"baseline/{ds_name}",
                save_visualizations=not args.no_viz,
                output_dir=viz_dir,
                max_viz_per_dataset=args.max_viz,
            )
            per_ds = metrics.get('per_dataset', {})
            if ds_name in per_ds:
                baseline_ds_metrics[ds_name] = per_ds[ds_name]
            else:
                baseline_ds_metrics[ds_name] = {k: 0.0 for k in METRIC_COLS}

        save_summary_txt(output_dir, 'baseline', baseline_ds_metrics)

    # ---- 保存对比 CSV ----
    save_comparison_csv(output_dir, trained_ds_metrics, baseline_ds_metrics)

    logger.info(f"全部测试完成！结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
