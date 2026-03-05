"""
统一的测试脚本
支持测试多种训练脚本的权重:
- train_onGen.py (gen_cffa, gen_cfoct, gen_octfa, gen_mixed)
- train_onReal.py (multiReal)
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

# 添加父目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scripts.v1.metrics import (
    compute_homography_errors,
    set_metrics_verbose,
    error_auc,
    compute_auc_rop
)

# 导入真实数据集
from data.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from data.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from data.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset


def is_valid_homography(H, scale_min=0.1, scale_max=10.0, perspective_threshold=0.005):
    """单应矩阵防爆锁"""
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


class RealDatasetWrapper(torch.utils.data.Dataset):
    """格式转换，用于真实数据验证"""
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

        # 转换为灰度图 [1, H, W]
        if fix_tensor.shape[0] == 3:
            fix_gray = 0.299 * fix_tensor[0] + 0.587 * fix_tensor[1] + 0.114 * fix_tensor[2]
            fix_gray = fix_gray.unsqueeze(0)
        else:
            fix_gray = fix_tensor

        if moving_gt_tensor.shape[0] == 3:
            moving_gray = 0.299 * moving_gt_tensor[0] + 0.587 * moving_gt_tensor[1] + 0.114 * moving_gt_tensor[2]
            moving_gray = moving_gray.unsqueeze(0)
        else:
            moving_gray = moving_gt_tensor

        if moving_original_tensor.shape[0] == 3:
            moving_orig_gray = 0.299 * moving_original_tensor[0] + 0.587 * moving_original_tensor[1] + 0.114 * moving_original_tensor[2]
            moving_orig_gray = moving_orig_gray.unsqueeze(0)
        else:
            moving_orig_gray = moving_original_tensor

        fix_name = os.path.basename(fix_path)
        moving_name = os.path.basename(moving_path)

        try:
            T_fix_to_moving = torch.inverse(T_0to1)
        except:
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


class TestDataModule:
    """测试用的数据模块，支持加载指定的数据集"""
    def __init__(self, args):
        self.args = args
        self.loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': True
        }

    def get_test_dataloader(self, datasets=None):
        """
        获取测试数据加载器
        datasets: list of dataset names to include, e.g., ['CFFA', 'CFOCT', 'OCTFA']
                 if None, load all datasets
        """
        script_dir = Path(__file__).parent.parent.parent
        val_dataset_list = []

        if datasets is None or 'CFFA' in datasets:
            cffa_dir = script_dir / 'data' / 'operation_pre_filtered_cffa'
            cffa_base = CFFADataset(root_dir=str(cffa_dir), split='val', mode='cf2fa')
            cffa_dataset = RealDatasetWrapper(cffa_base, split_name='test', dataset_name='CFFA')
            logger.info(f"加载 CFFA 测试集: {len(cffa_dataset)} 样本")
            val_dataset_list.append(cffa_dataset)

        if datasets is None or 'CFOCT' in datasets:
            cfoct_dir = script_dir / 'data' / 'operation_pre_filtered_cfoct'
            cfoct_base = CFOCTDataset(root_dir=str(cfoct_dir), split='val', mode='cf2oct')
            cfoct_dataset = RealDatasetWrapper(cfoct_base, split_name='test', dataset_name='CFOCT')
            logger.info(f"加载 CFOCT 测试集: {len(cfoct_dataset)} 样本")
            val_dataset_list.append(cfoct_dataset)

        if datasets is None or 'OCTFA' in datasets:
            octfa_dir = script_dir / 'data' / 'operation_pre_filtered_octfa'
            octfa_base = OCTFADataset(root_dir=str(octfa_dir), split='val', mode='fa2oct')
            octfa_dataset = RealDatasetWrapper(octfa_base, split_name='test', dataset_name='OCTFA')
            logger.info(f"加载 OCTFA 测试集: {len(octfa_dataset)} 样本")
            val_dataset_list.append(octfa_dataset)

        val_dataset = ConcatDataset(val_dataset_list)
        logger.info(f"测试集总样本数: {len(val_dataset)}")
        return DataLoader(val_dataset, shuffle=False, **self.loader_params)


def compute_metrics_for_dataset(evaluator, dataset_name):
    """为单个数据集计算指标"""
    if dataset_name in evaluator.per_dataset_errors and len(evaluator.per_dataset_errors[dataset_name]) > 0:
        errors = evaluator.per_dataset_errors[dataset_name]
        auc_dict = error_auc(errors, [5, 10, 20])
        mauc_dict = compute_auc_rop(errors, limit=25)

        mses = evaluator.per_dataset_mses.get(dataset_name, [])
        maces = evaluator.per_dataset_maces.get(dataset_name, [])

        return {
            'dataset': dataset_name,
            'auc@5': auc_dict.get('auc@5', 0.0),
            'auc@10': auc_dict.get('auc@10', 0.0),
            'auc@20': auc_dict.get('auc@20', 0.0),
            'mAUC': mauc_dict.get('mAUC', 0.0),
            'combined_auc': (auc_dict.get('auc@5', 0.0) + auc_dict.get('auc@10', 0.0) + auc_dict.get('auc@20', 0.0)) / 3.0,
            'mse': sum(mses) / len(mses) if mses else 0.0,
            'mace': sum(maces) / len(maces) if maces else 0.0,
            'num_samples': len(errors)
        }
    return None


class UnifiedEvaluator:
    """统一的评估器，支持按数据集分别计算指标"""
    def __init__(self, config=None):
        self.config = config
        self.reset()

    def reset(self):
        """重置累积的指标"""
        self.all_errors = []
        self.all_mses = []
        self.all_maces = []
        self.total_samples = 0
        self.failed_samples = 0
        self.inaccurate_samples = 0
        self.acceptable_samples = 0

        # 按数据集分别统计
        self.per_dataset_errors = {}
        self.per_dataset_mses = {}
        self.per_dataset_maces = {}
        self.per_dataset_samples = {}

    def evaluate_batch(self, batch, outputs, pl_module):
        """评估一个 batch"""
        matches0 = outputs['matches0']
        kpts0 = batch['keypoints0']
        kpts1 = batch['keypoints1']
        dataset_names = batch.get('dataset_name', ['unknown'] * kpts0.shape[0])

        B = kpts0.shape[0]

        mkpts0_f_list = []
        mkpts1_f_list = []
        m_bids_list = []

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

        # 按数据集分别统计
        for b in range(B):
            dataset = dataset_names[b] if isinstance(dataset_names, list) else dataset_names
            if dataset not in self.per_dataset_errors:
                self.per_dataset_errors[dataset] = []
                self.per_dataset_mses[dataset] = []
                self.per_dataset_maces[dataset] = []
                self.per_dataset_samples[dataset] = 0

            self.per_dataset_samples[dataset] += 1

            # 样本级别的误差
            if b < len(metrics_batch.get('t_errs', [])):
                self.per_dataset_errors[dataset].append(metrics_batch['t_errs'][b])

            # MSE 和 MACE
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
        """计算整个 epoch 的聚合指标"""
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

        # 计算 MSE 和 MACE
        valid_mses = [m for m in self.all_mses if np.isfinite(m)]
        valid_maces = [m for m in self.all_maces if np.isfinite(m)]
        metrics['mse'] = sum(valid_mses) / len(valid_mses) if valid_mses else 0.0
        metrics['mace'] = sum(valid_maces) / len(valid_maces) if valid_maces else 0.0
        metrics['inverse_mace'] = 1.0 / (1.0 + metrics['mace']) if metrics['mace'] > 0 else 1.0

        # 计算匹配失败率
        metrics['match_failure_rate'] = self.failed_samples / self.total_samples if self.total_samples > 0 else 0.0
        metrics['total_samples'] = self.total_samples
        metrics['failed_samples'] = self.failed_samples
        metrics['success_samples'] = self.total_samples - self.failed_samples
        metrics['inaccurate_samples'] = self.inaccurate_samples
        metrics['acceptable_samples'] = self.acceptable_samples
        metrics['inaccurate_rate'] = self.inaccurate_samples / self.total_samples if self.total_samples > 0 else 0.0
        metrics['acceptable_rate'] = self.acceptable_samples / self.total_samples if self.total_samples > 0 else 0.0

        # 按数据集分别计算指标
        metrics['per_dataset'] = {}
        for dataset_name in self.per_dataset_errors:
            dataset_metrics = compute_metrics_for_dataset(self, dataset_name)
            if dataset_metrics:
                metrics['per_dataset'][dataset_name] = dataset_metrics

        return metrics


def create_chessboard(img1, img2, grid_size=4):
    """创建棋盘格对比图"""
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


def run_evaluation(pl_module, dataloader, config=None, verbose=True, save_visualizations=False, output_dir=None):
    """
    运行完整的评估流程

    Args:
        pl_module: PyTorch Lightning 模块
        dataloader: 数据加载器
        config: 配置对象
        verbose: 是否输出详细日志
        save_visualizations: 是否保存可视化结果
        output_dir: 可视化结果保存目录

    Returns:
        dict: 评估指标
    """
    evaluator = UnifiedEvaluator(config=config)

    pl_module.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 将 batch 移到设备上
            batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # 前向传播
            outputs = pl_module(batch)

            # 评估
            result = evaluator.evaluate_batch(batch, outputs, pl_module)

            # 可视化
            if save_visualizations and output_dir:
                _visualize_batch(batch, result, output_dir, batch_idx)

            if verbose and batch_idx % 10 == 0:
                logger.info(f"已处理 {batch_idx + 1} 个 batch")

    # 计算聚合指标
    metrics = evaluator.compute_epoch_metrics()

    if verbose:
        logger.info(f"评估完成: 总样本数={metrics['total_samples']}, "
                   f"成功={metrics['success_samples']}, "
                   f"失败={metrics['failed_samples']}, "
                   f"不准确={metrics['inaccurate_samples']}, "
                   f"可接受={metrics['acceptable_samples']}")
        logger.info(f"MSE: {metrics['mse']:.6f}, MACE: {metrics['mace']:.4f}")
        logger.info(f"AUC@5: {metrics['auc@5']:.4f}, AUC@10: {metrics['auc@10']:.4f}, AUC@20: {metrics['auc@20']:.4f}")
        logger.info(f"Combined AUC: {metrics['combined_auc']:.4f}")

    return metrics


def _visualize_batch(batch, outputs, output_dir, batch_idx):
    """可视化一个batch的结果"""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    batch_size = batch['image0'].shape[0]
    H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)

    for sample_idx in range(batch_size):
        H_est = H_ests[sample_idx]

        if not is_valid_homography(H_est):
            H_est = np.eye(3)

        img0 = (batch['image0'][sample_idx, 0].cpu().numpy() * 255).astype(np.uint8)
        img1 = (batch['image1'][sample_idx, 0].cpu().numpy() * 255).astype(np.uint8)
        img1_gt = (batch['image1_gt'][sample_idx, 0].cpu().numpy() * 255).astype(np.uint8)

        h, w = img0.shape
        try:
            H_inv = np.linalg.inv(H_est)
            img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
        except:
            img1_result = img1.copy()

        # 构建样本名称
        pair_names = batch.get('pair_names', None)
        dataset_name = batch.get('dataset_name', ['unknown'] * batch_size)
        if isinstance(dataset_name, list):
            ds_name = dataset_name[sample_idx]
        else:
            ds_name = dataset_name

        if pair_names:
            sample_name = f"batch{batch_idx:04d}_sample{sample_idx:02d}_{ds_name}_{Path(pair_names[0][sample_idx]).stem}_vs_{Path(pair_names[1][sample_idx]).stem}"
        else:
            sample_name = f"batch{batch_idx:04d}_sample{sample_idx:02d}_{ds_name}"

        save_path = output_dir / sample_name
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存基本图像
        cv2.imwrite(str(save_path / "fix.png"), img0)
        cv2.imwrite(str(save_path / "moving_original.png"), img1)
        cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
        cv2.imwrite(str(save_path / "moving_gt.png"), img1_gt)

        # 绘制关键点和匹配
        img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

        if 'kpts0' in outputs and 'kpts1' in outputs:
            kpts0_np = outputs['kpts0'][sample_idx].cpu().numpy()
            kpts1_np = outputs['kpts1'][sample_idx].cpu().numpy()

            # 绘制所有关键点（白色）
            for pt in kpts0_np:
                cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
            for pt in kpts1_np:
                cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)

            # 绘制匹配点（红色）
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

                # 绘制匹配连线
                try:
                    margin = 10
                    h1, w1 = img0.shape[:2]
                    h2, w2 = img1.shape[:2]
                    H = max(h1, h2)
                    W = w1 + w2 + margin

                    out_img = np.zeros((H, W), dtype=np.uint8)
                    out_img[:h1, :w1] = img0
                    out_img[:h2, w1+margin:w1+margin+w2] = img1

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

        # 创建棋盘格对比图
        try:
            cb = create_chessboard(img1_result, img0)
            cv2.imwrite(str(save_path / "chessboard.png"), cb)
        except:
            pass

        # 保存单样本指标
        if 'mses' in outputs and sample_idx < len(outputs['mses']):
            mse = outputs['mses'][sample_idx]
            mace = outputs['maces'][sample_idx]
            with open(save_path / "metrics.txt", "w") as f:
                f.write(f"MSE: {mse:.6f}\n")
                f.write(f"MACE: {mace:.4f}\n")
                if 'matches0' in outputs:
                    m0 = outputs['matches0'][sample_idx].cpu()
                    valid = m0 > -1
                    num_matches = torch.sum(valid).item()
                    f.write(f"Matches: {num_matches}\n")


# ==========================================
# ==========================================
# 主函数
# ==========================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SuperGlue 统一测试脚本")
    parser.add_argument('--train_script', '-s', type=str, required=True,
                        choices=['train_onGen', 'train_onReal'],
                        help='训练脚本类型')
    parser.add_argument('--train_mode', '-m', type=str, default='mixed',
                        choices=['cffa', 'cfoct', 'octfa', 'mixed'],
                        help='训练模式')
    parser.add_argument('--test_datasets', '-d', type=str, default=None,
                        help='指定测试数据集，用逗号分隔，如 "CFFA,CFOCT,OCTFA" 或 "CFFA"')
    parser.add_argument('--name', '-n', type=str, required=True,
                        help='模型名称')
    parser.add_argument('--test_name', '-t', type=str, required=True,
                        help='测试名称')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='检查点路径')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--gpus', type=str, default='0', help='GPU设备ID')
    parser.add_argument('--img_size', type=int, default=512, help='图像大小')
    parser.add_argument('--no_viz', action='store_true', help='禁用可视化')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 导入必要的模块
    from scripts.v1_multi.train_onGen import PL_SuperGlue_Gen as GenModel
    from scripts.v1_multi.train_onReal import PL_SuperGlue_Real as RealModel
    from scripts.v1_multi.train_onGen import get_default_config as get_gen_config

    # 根据训练脚本选择模型类型
    if args.train_script == 'train_onGen':
        pl_class = GenModel
        mode_dir = f'gen_{args.train_mode}'
    else:
        pl_class = RealModel
        mode_dir = f'{args.train_mode}'

    # 获取配置
    config = get_gen_config()
    pl.seed_everything(config.TRAINER.SEED)

    # 确定检查点路径
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path(f"results/superglue_{mode_dir}/{args.name}/best_checkpoint/model.ckpt")

    if not ckpt_path.exists():
        logger.error(f"未找到检查点: {ckpt_path}")
        return

    logger.info(f"加载检查点: {ckpt_path}")

    # 设置输出目录
    output_dir = Path(f"results/superglue_{mode_dir}/{args.name}/{args.test_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 配置日志
    log_file = output_dir / "test_log.txt"
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level="INFO")
    logger.add(log_file, format=log_format, level="INFO", mode="w")
    logger.info(f"日志将保存到: {log_file}")

    # GPU配置
    if ',' in str(args.gpus):
        gpus_list = [int(x) for x in args.gpus.split(',')]
        _n_gpus = len(gpus_list)
    else:
        try:
            gpus_list = [int(args.gpus)]
            _n_gpus = 1
        except:
            gpus_list = 'auto'
            _n_gpus = 1

    # 确定测试数据集（与 LightGlue 对齐）
    # 如果指定了 --test_datasets 则使用指定的，否则根据 train_mode 自动选择
    if args.test_datasets:
        test_datasets_list = [ds.strip() for ds in args.test_datasets.split(',')]
        logger.info(f"指定测试数据集: {test_datasets_list}")
    else:
        # 根据 train_mode 自动选择对应数据集
        mode2datasets = {
            'cffa': ['CFFA'],
            'cfoct': ['CFOCT'],
            'octfa': ['OCTFA'],
            'mixed': ['CFFA', 'CFOCT', 'OCTFA']
        }
        test_datasets_list = mode2datasets.get(args.train_mode, ['CFFA', 'CFOCT', 'OCTFA'])
        logger.info(f"根据 train_mode 自动选择测试数据集: {test_datasets_list}")

    logger.info(f"训练脚本: {args.train_script}")
    logger.info(f"训练模式: {args.train_mode}")
    logger.info(f"测试数据集: {test_datasets_list}")
    logger.info(f"模型名称: {args.name}")
    logger.info(f"测试名称: {args.test_name}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"GPU配置: devices={gpus_list}, num_gpus={_n_gpus}")

    # 从检查点加载模型
    model = pl_class.load_from_checkpoint(
        str(ckpt_path),
        config=config,
        result_dir=str(output_dir)
    )
    model.eval()

    # 初始化数据模块
    test_data_module = TestDataModule(args)
    test_dataloader = test_data_module.get_test_dataloader(datasets=test_datasets_list)

    logger.info(f"开始测试 (数据集: {test_datasets_list} | 模型: {args.name})")

    # 使用 run_evaluation 函数进行测试
    set_metrics_verbose(True)
    save_viz = not args.no_viz
    metrics = run_evaluation(
        model,
        test_dataloader,
        config=config,
        verbose=True,
        save_visualizations=save_viz,
        output_dir=output_dir
    )

    # 保存测试总结
    summary_path = output_dir / "test_summary.txt"
    with open(summary_path, "w") as f:
        f.write("测试总结\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model Name: {args.name}\n")
        f.write(f"Test Name: {args.test_name}\n")
        f.write(f"Test Datasets: {test_datasets_list}\n")
        f.write("-" * 50 + "\n")
        f.write(f"总样本数: {metrics['total_samples']}\n")
        f.write(f"匹配成功样本数: {metrics['success_samples']}\n")
        f.write(f"匹配失败样本数: {metrics['failed_samples']}\n")
        f.write(f"匹配失败率: {metrics['match_failure_rate']:.4f}\n")
        f.write(f"Inaccurate 样本数: {metrics['inaccurate_samples']}\n")
        f.write(f"Inaccurate 率: {metrics['inaccurate_rate']:.4f}\n")
        f.write(f"Acceptable 样本数: {metrics['acceptable_samples']}\n")
        f.write(f"Acceptable 率: {metrics['acceptable_rate']:.4f}\n")
        f.write(f"MSE (仅匹配成功): {metrics['mse']:.6f}\n")
        f.write(f"MACE (仅匹配成功): {metrics['mace']:.4f}\n")
        f.write(f"AUC@5: {metrics['auc@5']:.4f}\n")
        f.write(f"AUC@10: {metrics['auc@10']:.4f}\n")
        f.write(f"AUC@20: {metrics['auc@20']:.4f}\n")
        f.write(f"mAUC: {metrics['mAUC']:.4f}\n")
        f.write(f"Combined AUC: {metrics['combined_auc']:.4f}\n")
        f.write(f"Inverse MACE: {metrics['inverse_mace']:.6f}\n")

        # 写入每个数据集的单独指标
        if 'per_dataset' in metrics and metrics['per_dataset']:
            f.write("\n按数据集统计:\n")
            f.write("-" * 50 + "\n")
            for ds_name, ds_metrics in metrics['per_dataset'].items():
                f.write(f"\n{ds_name}:\n")
                f.write(f"  样本数: {ds_metrics['num_samples']}\n")
                f.write(f"  AUC@5: {ds_metrics['auc@5']:.4f}\n")
                f.write(f"  AUC@10: {ds_metrics['auc@10']:.4f}\n")
                f.write(f"  AUC@20: {ds_metrics['auc@20']:.4f}\n")
                f.write(f"  mAUC: {ds_metrics['mAUC']:.4f}\n")
                f.write(f"  Combined AUC: {ds_metrics['combined_auc']:.4f}\n")
                f.write(f"  MSE: {ds_metrics['mse']:.6f}\n")
                f.write(f"  MACE: {ds_metrics['mace']:.4f}\n")

    logger.info(f"测试总结已保存到: {summary_path}")
    logger.info(f"测试完成! 结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
