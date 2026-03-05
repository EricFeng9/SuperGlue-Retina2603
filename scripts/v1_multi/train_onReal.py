import sys
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pprint
from pathlib import Path
from loguru import logger
import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import ConcatDataset
import logging
from types import SimpleNamespace

# 添加父目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 导入 SuperGlue 和 SuperPoint（本地models）
from models.superglue import SuperGlue
from models.superpoint import SuperPoint

# 导入数据集
from data.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from data.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from data.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset

# 导入统一的测试/验证模块
from scripts.v1.test import UnifiedEvaluator

# ==========================================
# 配置函数
# ==========================================
def get_default_config():
    conf = SimpleNamespace()
    conf.TRAINER = SimpleNamespace()
    conf.TRAINER.CANONICAL_BS = 4
    conf.TRAINER.CANONICAL_LR = 1e-4
    conf.TRAINER.TRUE_LR = 1e-4
    conf.TRAINER.RANSAC_PIXEL_THR = 3.0
    conf.TRAINER.SEED = 66
    conf.TRAINER.WORLD_SIZE = 1
    conf.TRAINER.TRUE_BATCH_SIZE = 4
    conf.TRAINER.PLOT_MODE = 'evaluation'
    conf.TRAINER.PATIENCE = 10

    # SuperPoint 配置
    conf.SUPERPOINT = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 2048,
        'remove_borders': 4,
    }

    conf.SUPERPOINT_PRETRAINED = None  # 设置为 superpoint_v1.pth 的路径

    # SuperGlue 配置
    conf.SUPERGLUE = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    return conf

# ==========================================
# 工具函数
# ==========================================
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

def filter_valid_area(img1, img2):
    assert img1.shape[:2] == img2.shape[:2]
    if len(img1.shape) == 3:
        mask1 = np.any(img1 > 10, axis=2)
    else:
        mask1 = img1 > 0
    if len(img2.shape) == 3:
        mask2 = np.any(img2 > 10, axis=2)
    else:
        mask2 = img2 > 0
    valid_mask = mask1 & mask2
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return img1, img2
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    filtered_img1 = img1[row_min:row_max+1, col_min:col_max+1].copy()
    filtered_img2 = img2[row_min:row_max+1, col_min:col_max+1].copy()
    valid_mask_cropped = valid_mask[row_min:row_max+1, col_min:col_max+1]
    filtered_img1[~valid_mask_cropped] = 0
    filtered_img2[~valid_mask_cropped] = 0
    return filtered_img1, filtered_img2

def compute_corner_error(H_est, H_gt, height, width):
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    corners_homo = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
    corners_gt_homo = (H_gt @ corners_homo.T).T
    corners_gt = corners_gt_homo[:, :2] / (corners_gt_homo[:, 2:] + 1e-6)
    corners_est_homo = (H_est @ corners_homo.T).T
    corners_est = corners_est_homo[:, :2] / (corners_est_homo[:, 2:] + 1e-6)
    try:
        errors = np.sqrt(np.sum((corners_est - corners_gt)**2, axis=1))
        mace = np.mean(errors)
    except:
        mace = float('inf')
    return mace

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

# ==========================================
# 数据集包装类
# ==========================================
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

class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': True
        }

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            script_dir = Path(__file__).parent.parent.parent

            # 获取训练和验证数据集列表
            train_datasets = getattr(self.args, 'train_datasets', ['CFFA', 'CFOCT', 'OCTFA'])
            val_datasets = getattr(self.args, 'val_datasets', ['CFFA', 'CFOCT', 'OCTFA'])

            # 训练集
            train_dataset_list = []
            if 'CFFA' in train_datasets:
                cffa_dir = script_dir / 'data' / 'operation_pre_filtered_cffa'
                cffa_base = CFFADataset(root_dir=str(cffa_dir), split='train', mode='cf2fa')
                cffa_dataset = RealDatasetWrapper(cffa_base, split_name='train', dataset_name='CFFA')
                logger.info(f"训练集加载 CFFA: {len(cffa_dataset)} 样本")
                train_dataset_list.append(cffa_dataset)

            if 'CFOCT' in train_datasets:
                cfoct_dir = script_dir / 'data' / 'operation_pre_filtered_cfoct'
                cfoct_base = CFOCTDataset(root_dir=str(cfoct_dir), split='train', mode='cf2oct')
                cfoct_dataset = RealDatasetWrapper(cfoct_base, split_name='train', dataset_name='CFOCT')
                logger.info(f"训练集加载 CFOCT: {len(cfoct_dataset)} 样本")
                train_dataset_list.append(cfoct_dataset)

            if 'OCTFA' in train_datasets:
                octfa_dir = script_dir / 'data' / 'operation_pre_filtered_octfa'
                octfa_base = OCTFADataset(root_dir=str(octfa_dir), split='train', mode='fa2oct')
                octfa_dataset = RealDatasetWrapper(octfa_base, split_name='train', dataset_name='OCTFA')
                logger.info(f"训练集加载 OCTFA: {len(octfa_dataset)} 样本")
                train_dataset_list.append(octfa_dataset)

            self.train_dataset = ConcatDataset(train_dataset_list)
            logger.info(f"训练集总样本数: {len(self.train_dataset)}")

            # 验证集
            val_dataset_list = []
            if 'CFFA' in val_datasets:
                cffa_dir = script_dir / 'data' / 'operation_pre_filtered_cffa'
                cffa_base = CFFADataset(root_dir=str(cffa_dir), split='val', mode='cf2fa')
                cffa_dataset = RealDatasetWrapper(cffa_base, split_name='test', dataset_name='CFFA')
                logger.info(f"验证集加载 CFFA: {len(cffa_dataset)} 样本")
                val_dataset_list.append(cffa_dataset)

            if 'CFOCT' in val_datasets:
                cfoct_dir = script_dir / 'data' / 'operation_pre_filtered_cfoct'
                cfoct_base = CFOCTDataset(root_dir=str(cfoct_dir), split='val', mode='cf2oct')
                cfoct_dataset = RealDatasetWrapper(cfoct_base, split_name='test', dataset_name='CFOCT')
                logger.info(f"验证集加载 CFOCT: {len(cfoct_dataset)} 样本")
                val_dataset_list.append(cfoct_dataset)

            if 'OCTFA' in val_datasets:
                octfa_dir = script_dir / 'data' / 'operation_pre_filtered_octfa'
                octfa_base = OCTFADataset(root_dir=str(octfa_dir), split='val', mode='fa2oct')
                octfa_dataset = RealDatasetWrapper(octfa_base, split_name='test', dataset_name='OCTFA')
                logger.info(f"验证集加载 OCTFA: {len(octfa_dataset)} 样本")
                val_dataset_list.append(octfa_dataset)

            self.val_dataset = ConcatDataset(val_dataset_list)
            logger.info(f"验证集总样本数: {len(self.val_dataset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, **self.loader_params)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle=False, **self.loader_params)

# ==========================================
# 模型类: PL_SuperGlue_Real
# ==========================================
class PL_SuperGlue_Real(pl.LightningModule):
    def __init__(self, config, result_dir=None):
        super().__init__()
        self.config = config
        self.result_dir = result_dir
        self.save_hyperparameters({'config': str(config)})

        # 1. 特征提取器 (SuperPoint) - 冻结，使用预训练权重
        sp_config = config.SUPERPOINT.copy()
        pretrained_path = config.SUPERPOINT_PRETRAINED
        self.extractor = SuperPoint(sp_config, pretrained_path=pretrained_path).eval()
        for param in self.extractor.parameters():
            param.requires_grad = False

        # 2. 匹配器 (SuperGlue) - 可训练
        sg_conf = config.SUPERGLUE.copy()
        self.matcher = SuperGlue(sg_conf)

        self.force_viz = False

        # 使用统一的评估器
        self.evaluator = UnifiedEvaluator(mode='real', config=config)

    def configure_optimizers(self):
        lr = self.config.TRAINER.TRUE_LR
        optimizer = torch.optim.Adam(self.matcher.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=self.config.TRAINER.PATIENCE, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "combined_auc",
                "strict": False,
            },
        }

    def forward(self, batch):
        with torch.no_grad():
            if 'keypoints0' not in batch:
                feats0 = self.extractor({'image': batch['image0']})
                feats1 = self.extractor({'image': batch['image1']})

                def list_to_batch(feats):
                    lengths = [int(k.shape[0]) for k in feats['keypoints']]
                    if len(lengths) == 0:
                        raise ValueError('Empty SuperPoint features.')
                    n_common = min(lengths)
                    if n_common <= 0:
                        device = feats['descriptors'][0].device if len(feats.get('descriptors', [])) else batch['image0'].device
                        keypoints = torch.zeros((len(lengths), 0, 2), device=device, dtype=torch.float32)
                        descriptors = torch.zeros((len(lengths), 256, 0), device=device, dtype=torch.float32)
                        scores = torch.zeros((len(lengths), 0), device=device, dtype=torch.float32)
                        return keypoints, descriptors, scores

                    kpts_b, desc_b, scores_b = [], [], []
                    for k, d, s in zip(feats['keypoints'], feats['descriptors'], feats['scores']):
                        k = k.float()
                        d = d.float()
                        s = s.float()
                        if k.shape[0] > n_common:
                            topk = torch.topk(s, k=n_common, dim=0, largest=True, sorted=False).indices
                            k = k[topk]
                            s = s[topk]
                            d = d[:, topk]
                        kpts_b.append(k)
                        desc_b.append(d)
                        scores_b.append(s)
                    keypoints = torch.stack(kpts_b, dim=0)
                    descriptors = torch.stack(desc_b, dim=0)
                    scores = torch.stack(scores_b, dim=0)
                    return keypoints, descriptors, scores

                kpts0, desc0, sc0 = list_to_batch(feats0)
                kpts1, desc1, sc1 = list_to_batch(feats1)

                batch.update({
                    'keypoints0': kpts0,
                    'descriptors0': desc0,
                    'scores0': sc0,
                    'keypoints1': kpts1,
                    'descriptors1': desc1,
                    'scores1': sc1
                })

        data = {
            'descriptors0': batch['descriptors0'],
            'descriptors1': batch['descriptors1'],
            'keypoints0': batch['keypoints0'],
            'keypoints1': batch['keypoints1'],
            'scores0': batch['scores0'],
            'scores1': batch['scores1'],
            'image0': batch['image0'],
            'image1': batch['image1']
        }

        return self.matcher(data)

    def forward_with_scores(self, batch):
        """前向传播，返回用于训练的分数"""
        with torch.no_grad():
            if 'keypoints0' not in batch:
                feats0 = self.extractor({'image': batch['image0']})
                feats1 = self.extractor({'image': batch['image1']})

                def list_to_batch(feats):
                    lengths = [int(k.shape[0]) for k in feats['keypoints']]
                    if len(lengths) == 0:
                        raise ValueError('Empty SuperPoint features.')
                    n_common = min(lengths)
                    if n_common <= 0:
                        device = feats['descriptors'][0].device if len(feats.get('descriptors', [])) else batch['image0'].device
                        keypoints = torch.zeros((len(lengths), 0, 2), device=device, dtype=torch.float32)
                        descriptors = torch.zeros((len(lengths), 256, 0), device=device, dtype=torch.float32)
                        scores = torch.zeros((len(lengths), 0), device=device, dtype=torch.float32)
                        return keypoints, descriptors, scores

                    kpts_b, desc_b, scores_b = [], [], []
                    for k, d, s in zip(feats['keypoints'], feats['descriptors'], feats['scores']):
                        k = k.float()
                        d = d.float()
                        s = s.float()
                        if k.shape[0] > n_common:
                            topk = torch.topk(s, k=n_common, dim=0, largest=True, sorted=False).indices
                            k = k[topk]
                            s = s[topk]
                            d = d[:, topk]
                        kpts_b.append(k)
                        desc_b.append(d)
                        scores_b.append(s)
                    keypoints = torch.stack(kpts_b, dim=0)
                    descriptors = torch.stack(desc_b, dim=0)
                    scores = torch.stack(scores_b, dim=0)
                    return keypoints, descriptors, scores

                kpts0, desc0, sc0 = list_to_batch(feats0)
                kpts1, desc1, sc1 = list_to_batch(feats1)

                batch.update({
                    'keypoints0': kpts0,
                    'descriptors0': desc0,
                    'scores0': sc0,
                    'keypoints1': kpts1,
                    'descriptors1': desc1,
                    'scores1': sc1
                })

        data = {
            'descriptors0': batch['descriptors0'],
            'descriptors1': batch['descriptors1'],
            'keypoints0': batch['keypoints0'],
            'keypoints1': batch['keypoints1'],
            'scores0': batch['scores0'],
            'scores1': batch['scores1'],
            'image0': batch['image0'],
            'image1': batch['image1']
        }

        return self.matcher(data, return_scores=True)

    def _compute_gt_matches(self, kpts0, kpts1, T_0to1, dist_th=3.0):
        B, M, _ = kpts0.shape
        B, N, _ = kpts1.shape
        device = kpts0.device

        kpts0_h = torch.cat([kpts0, torch.ones(B, M, 1, device=device)], dim=-1)
        kpts0_warped_h = torch.matmul(kpts0_h, T_0to1.transpose(1, 2))
        kpts0_warped = kpts0_warped_h[..., :2] / (kpts0_warped_h[..., 2:] + 1e-8)

        dist = torch.cdist(kpts0_warped, kpts1)
        min_dist, matched_indices = torch.min(dist, dim=-1)
        mask = min_dist < dist_th
        matches_gt = torch.where(mask, matched_indices, torch.tensor(-1, device=device))

        return matches_gt

    def _compute_loss(self, outputs, kpts0, kpts1, T_0to1):
        # SuperGlue 返回的 scores 格式: [B, M+1, N+1]
        scores = outputs['scores']
        matches_gt = self._compute_gt_matches(kpts0, kpts1, T_0to1)

        B, M, N = scores.shape[0], scores.shape[1] - 1, scores.shape[2] - 1

        targets = matches_gt.clone()
        targets[targets == -1] = N

        target_log_probs = torch.gather(scores[:, :M, :], 2, targets.unsqueeze(2)).squeeze(2)
        loss = -target_log_probs.mean()

        return loss

    def training_step(self, batch, batch_idx):
        outputs = self.forward_with_scores(batch)
        loss = self._compute_loss(outputs, batch['keypoints0'], batch['keypoints1'], batch['T_0to1'])

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """验证步骤（使用统一的评估器）"""
        outputs = self(batch)

        outputs_scores = self.forward_with_scores(batch)
        loss = self._compute_loss(outputs_scores, batch['keypoints0'], batch['keypoints1'], batch['T_0to1'])
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # 使用统一的评估器
        result = self.evaluator.evaluate_batch(batch, outputs, self)

        return result

    def on_validation_epoch_start(self):
        """每个验证 epoch 开始时重置评估器"""
        self.evaluator.reset()

    def on_validation_epoch_end(self):
        """在模型自身 hook 中 log combined_auc，确保 EarlyStopping 能找到该指标"""
        # 使用统一的评估器计算聚合指标
        metrics = self.evaluator.compute_epoch_metrics()

        # Log 所有指标
        self.log('auc@5',        metrics['auc@5'],        on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('auc@10',       metrics['auc@10'],       on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('auc@20',       metrics['auc@20'],       on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('mAUC',         metrics.get('mAUC', 0.0), on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('combined_auc', metrics['combined_auc'], on_epoch=True, prog_bar=True,  logger=True, sync_dist=False)
        self.log('val_mse',      metrics['mse'],          on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('val_mace',     metrics['mace'],         on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('inverse_mace', metrics['inverse_mace'], on_epoch=True, prog_bar=False, logger=True, sync_dist=False)

# ==========================================
# 回调类: MultimodalValidationCallback
# ==========================================
class MultimodalValidationCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.best_val = -1.0
        self.result_dir = Path(f"results/superglue_{args.mode}/{args.name}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_mses = []
        self.epoch_maces = []

        import csv
        self.csv_path = self.result_dir / "metrics.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val MSE", "Val MACE", "Val AUC@5", "Val AUC@10", "Val AUC@20", "Val Combined AUC", "Val Inverse MACE"])

        self.current_train_metrics = {}
        self.current_val_metrics = {}

    def _try_write_csv(self, epoch):
        if epoch in self.current_train_metrics and epoch in self.current_val_metrics:
            t = self.current_train_metrics.pop(epoch)
            v = self.current_val_metrics.pop(epoch)
            import csv
            with open(self.csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    t.get('loss', ''),
                    v.get('val_loss', ''),
                    v['mse'],
                    v['mace'],
                    v['auc5'],
                    v['auc10'],
                    v['auc20'],
                    v['combined_auc'],
                    v['inverse_mace']
                ])

    def on_validation_epoch_start(self, trainer, pl_module):
        self.epoch_mses = []
        self.epoch_maces = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if 'mses' in outputs:
            self.epoch_mses.extend(outputs['mses'])
        if 'maces' in outputs:
            self.epoch_maces.extend(outputs['maces'])

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        display_metrics = {}

        if 'train/loss_epoch' in metrics:
            display_metrics['loss'] = metrics['train/loss_epoch'].item()

        if display_metrics:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
            logger.info(f"Epoch {epoch} 训练总结 >> {metric_str}")

        self.current_train_metrics[epoch] = display_metrics
        self._try_write_csv(epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics

        display_metrics = {}
        for k in ['val_loss', 'val_mse', 'val_mace', 'auc@5', 'auc@10', 'auc@20', 'mAUC', 'combined_auc', 'inverse_mace']:
            if k in metrics:
                display_metrics[k] = metrics[k].item()

        auc5 = display_metrics.get('auc@5', 0.0)
        auc10 = display_metrics.get('auc@10', 0.0)
        auc20 = display_metrics.get('auc@20', 0.0)
        combined_auc = display_metrics.get('combined_auc', 0.0)
        avg_mse = display_metrics.get('val_mse', 0.0)
        avg_mace = display_metrics.get('val_mace', 0.0)
        inverse_mace = display_metrics.get('inverse_mace', 0.0)

        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
        logger.info(f"Epoch {epoch} 验证总结 >> {metric_str}")

        self.current_val_metrics[epoch] = {
            'mse': avg_mse,
            'mace': avg_mace,
            'auc5': auc5,
            'auc10': auc10,
            'auc20': auc20,
            'combined_auc': combined_auc,
            'inverse_mace': inverse_mace,
            'val_loss': display_metrics.get('val_loss', 0.0)
        }
        self._try_write_csv(epoch)

        latest_path = self.result_dir / "latest_checkpoint"
        latest_path.mkdir(exist_ok=True)
        trainer.save_checkpoint(latest_path / "model.ckpt")

        is_best = False
        if combined_auc > self.best_val:
            self.best_val = combined_auc
            is_best = True
            best_path = self.result_dir / "best_checkpoint"
            best_path.mkdir(exist_ok=True)
            trainer.save_checkpoint(best_path / "model.ckpt")
            with open(best_path / "log.txt", "w") as f:
                f.write(f"Epoch: {epoch}\nBest Combined AUC: {combined_auc:.4f}\n")
                f.write(f"AUC@5: {auc5:.4f}\nAUC@10: {auc10:.4f}\nAUC@20: {auc20:.4f}\n")
                f.write(f"MACE: {avg_mace:.4f}\nMSE: {avg_mse:.6f}\n")
            logger.info(f"发现新的最优模型! Epoch {epoch}, Combined AUC: {combined_auc:.4f}")

        if is_best or (epoch % 5 == 0):
            self._trigger_visualization(trainer, pl_module, is_best, epoch)

    def _trigger_visualization(self, trainer, pl_module, is_best, epoch):
        pl_module.force_viz = True
        target_dir = self.result_dir / (f"epoch{epoch}_best" if is_best else f"epoch{epoch}")
        target_dir.mkdir(parents=True, exist_ok=True)

        val_dataloader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
        pl_module.eval()

        visualized_count = 0
        max_visualize = 20

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = pl_module.validation_step(batch, batch_idx)

                batch_size = batch['image0'].shape[0]
                splits = batch.get('split', ['unknown'] * batch_size)

                for i in range(batch_size):
                    sample_split = splits[i] if isinstance(splits, list) else splits

                    if sample_split == 'test':
                        self._process_batch_sample(trainer, pl_module, batch, outputs, target_dir, i, batch_idx, sample_split)
                        visualized_count += 1

                        if visualized_count >= max_visualize:
                            break

                if visualized_count >= max_visualize:
                    break

        logger.info(f"已可视化 {visualized_count} 个测试集样本")
        pl_module.force_viz = False

    def _process_batch_sample(self, trainer, pl_module, batch, outputs, epoch_dir, sample_idx, batch_idx, split):
        H_ests = outputs.get('H_est', [np.eye(3)] * batch['image0'].shape[0])
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

        sample_name = f"batch{batch_idx:04d}_sample{sample_idx:02d}_{split}_{Path(batch['pair_names'][0][sample_idx]).stem}_vs_{Path(batch['pair_names'][1][sample_idx]).stem}"
        save_path = epoch_dir / sample_name
        save_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path / "fix.png"), img0)
        cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
        cv2.imwrite(str(save_path / "moving_gt.png"), img1_gt)

        img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

        matches_info = None

        if 'kpts0' in outputs and 'kpts1' in outputs:
            kpts0_np = outputs['kpts0'][sample_idx].cpu().numpy() if hasattr(outputs['kpts0'][sample_idx], 'cpu') else batch['keypoints0'][sample_idx].cpu().numpy()
            kpts1_np = outputs['kpts1'][sample_idx].cpu().numpy() if hasattr(outputs['kpts1'][sample_idx], 'cpu') else batch['keypoints1'][sample_idx].cpu().numpy()

            valid_mask_0 = (kpts0_np[:, 0] != 0) | (kpts0_np[:, 1] != 0)
            valid_mask_1 = (kpts1_np[:, 0] != 0) | (kpts1_np[:, 1] != 0)
            kpts0_valid = kpts0_np[valid_mask_0]
            kpts1_valid = kpts1_np[valid_mask_1]

            for pt in kpts0_valid:
                cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
            for pt in kpts1_valid:
                cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)

            if 'matches0' in outputs:
                m0 = outputs['matches0'][sample_idx].cpu() if hasattr(outputs['matches0'][sample_idx], 'cpu') else outputs['matches0'][sample_idx]
                valid = m0 > -1
                m_indices_0 = torch.where(valid)[0].numpy()
                m_indices_1 = m0[valid].numpy()

                matches_info = []
                for idx0, idx1 in zip(m_indices_0, m_indices_1):
                    if idx0 < len(kpts0_valid) and idx1 < len(kpts1_valid):
                        pt0 = kpts0_valid[idx0]
                        pt1 = kpts1_valid[idx1]
                        matches_info.append((pt0, pt1))
                        cv2.circle(img0_color, (int(pt0[0]), int(pt0[1])), 4, (0, 0, 255), -1)
                        cv2.circle(img1_color, (int(pt1[0]), int(pt1[1])), 4, (0, 0, 255), -1)

        cv2.imwrite(str(save_path / "fix_with_kpts.png"), img0_color)
        cv2.imwrite(str(save_path / "moving_with_kpts.png"), img1_color)

        if matches_info:
            h, w = img0.shape
            matches_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
            matches_img[:, :w] = img0_color
            matches_img[:, w:] = img1_color

            for pt0, pt1 in matches_info:
                x0, y0 = int(pt0[0]), int(pt0[1])
                x1, y1 = int(pt1[0]) + w, int(pt1[1])
                cv2.line(matches_img, (x0, y0), (x1, y1), (0, 255, 0), 1)

            cv2.imwrite(str(save_path / "matches.png"), matches_img)

        try:
            cb = create_chessboard(img1_result, img0)
            cv2.imwrite(str(save_path / "chessboard.png"), cb)
        except:
            pass

# ==========================================
# 早停机制
# ==========================================
class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)

# ==========================================
# 主函数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="SuperGlue Multi-Modal Real-Data Training")
    parser.add_argument('--name', '-n', type=str, default='superglue_multireal', help='训练名称')
    parser.add_argument('--mode', type=str, default='cffa', choices=['cffa', 'cfoct', 'octfa'], help='数据集模式: cffa, cfoct 或 octfa')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--start_point', type=str, default=None, help='从检查点恢复')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--gpus', type=str, default='1')

    # SuperPoint 预训练权重路径
    parser.add_argument('--superpoint_pretrained', type=str, default=None,
                        help='SuperPoint 预训练权重路径 (默认从 models/weights/ 加载)')

    return parser.parse_args()

def main():
    args = parse_args()

    # 根据 mode 参数设置训练和验证数据集（与 LightGlue 对齐）
    # train_onReal 只支持单个数据集训练
    train_datasets_config = {
        'cffa': ['CFFA'],
        'cfoct': ['CFOCT'],
        'octfa': ['OCTFA']
    }
    val_datasets_config = {
        'cffa': ['CFFA'],
        'cfoct': ['CFOCT'],
        'octfa': ['OCTFA']
    }

    args.train_datasets = train_datasets_config.get(args.mode, ['CFFA'])
    args.val_datasets = val_datasets_config.get(args.mode, ['CFFA'])

    config = get_default_config()
    config.TRAINER.SEED = 66
    pl.seed_everything(config.TRAINER.SEED)

    # 设置 SuperPoint 预训练权重路径
    if args.superpoint_pretrained:
        config.SUPERPOINT_PRETRAINED = args.superpoint_pretrained
    else:
        default_sp_path = Path(__file__).parent.parent / 'models' / 'weights' / 'superpoint_v1.pth'
        if default_sp_path.exists():
            config.SUPERPOINT_PRETRAINED = str(default_sp_path)
            logger.info(f"使用默认 SuperPoint 预训练权重: {default_sp_path}")
        else:
            logger.warning(f"未找到 SuperPoint 预训练权重: {default_sp_path}，将使用随机初始化")

    result_dir = Path(f"results/superglue_{args.mode}/{args.name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = result_dir / "log.txt"

    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level="INFO")
    logger.add(log_file, format=log_format, level="INFO", mode="a", backtrace=True, diagnose=False)
    logger.info(f"日志将同时保存到: {log_file}")

    os.environ['LOFTR_LOG_FILE'] = str(log_file)

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

    config.TRAINER.WORLD_SIZE = max(_n_gpus, 1)
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling

    model = PL_SuperGlue_Real(config, result_dir=str(result_dir))
    data_module = MultimodalDataModule(args, config)

    tb_logger = TensorBoardLogger(save_dir='logs/tb_logs', name=f"superglue_{args.name}")

    early_stop_callback = DelayedEarlyStopping(
        start_epoch=0,
        monitor='combined_auc',
        mode='max',
        patience=10,
        min_delta=0.0001,
        strict=False
    )

    logger.info("=" * 80)
    logger.info("【SuperGlue 多模态真实数据训练】")
    logger.info("=" * 80)
    logger.info(f"训练数据集: {args.train_datasets}")
    logger.info(f"验证数据集: {args.val_datasets}")
    logger.info(f"SuperPoint 预训练权重: {config.SUPERPOINT_PRETRAINED}")
    logger.info(f"早停配置: monitor=combined_auc, start_epoch=0, patience=10, min_delta=0.0001")
    logger.info("=" * 80)

    logger.info(f"GPU 配置: devices={gpus_list}, num_gpus={_n_gpus}")
    logger.info(f"学习率: {config.TRAINER.TRUE_LR:.6f} (scaled from {config.TRAINER.CANONICAL_LR})")

    trainer_kwargs = {
        'max_epochs': args.max_epochs,
        'accelerator': "gpu" if torch.cuda.is_available() else "cpu",
        'devices': gpus_list,
        'num_sanity_val_steps': 0,
        'check_val_every_n_epoch': 1,
        'callbacks': [
            MultimodalValidationCallback(args),
            LearningRateMonitor(logging_interval='step'),
            early_stop_callback
        ],
        'logger': tb_logger,
    }

    if _n_gpus > 1:
        trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(**trainer_kwargs)

    ckpt_path = args.start_point if args.start_point else None

    logger.info(f"开始多模态真实数据训练: {args.name}")
    logger.info(f"训练数据集: {args.train_datasets}")
    logger.info(f"验证数据集: {args.val_datasets}")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()
