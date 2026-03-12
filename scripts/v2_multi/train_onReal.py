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
from torch.utils.data._utils.collate import default_collate
import logging
from types import SimpleNamespace

# 添加父目录到 sys.path
# 先添加 SuperGlue 目录，以便导入 superglue 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
# 再添加项目根目录，以便导入 dataset 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# 导入 SuperGlue 相关模块
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from models import matching

# 导入数据集（使用与 test.py 一致的数据集）
from dataset.CFFA.cffa_dataset import CFFADataset
from dataset.CF_OCT.cf_oct_dataset import CFOCTDataset
from dataset.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset

# 导入统一的测试/验证模块（使用 v2_multi 版本，与 metrics 保持一致）
from scripts.v2_multi.test import UnifiedEvaluator

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
    conf.TRAINER.PATIENCE = 10  # 默认 patience 值
    
    conf.MATCHING = {
        'superpoint': {
            'descriptor_dim': 256,
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 2048,
            'remove_borders': 4,
        },
        'superglue': {
            'descriptor_dim': 256,
            'weights': 'indoor',
            'keypoint_encoder': [32, 64, 128, 256],
            'GNN_layers': ['self', 'cross'] * 9,
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
        },
    }
    
    # SuperPoint 预训练权重路径
    conf.SUPERPOINT_PRETRAINED = None  # None 表示从默认路径加载（如果不存在会自动下载）
    # SuperGlue 预训练权重路径
    conf.SUPERGLUE_PRETRAINED = None  # None 表示从默认路径加载 indoor/outdoor 权重
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

def real_batch_collate(batch):
    """Collate batch for RealDatasetWrapper: gt_pts0/gt_pts1 变长不 stack，保持为 list；pair_names 转为 (list_fix, list_mov)。"""
    if not batch:
        return {}
    first = batch[0]
    collated = {}
    for k in first.keys():
        if k == 'gt_pts0' or k == 'gt_pts1':
            collated[k] = [sample[k] for sample in batch]
        elif k == 'pair_names':
            collated[k] = ([sample[k][0] for sample in batch], [sample[k][1] for sample in batch])
        else:
            collated[k] = default_collate([sample[k] for sample in batch])
    return collated

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
        self.target_size = 512
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        
        # 获取原始样本（包含GT关键点）
        fix_points = np.array([], dtype=np.float32).reshape(0, 2)
        moving_points = np.array([], dtype=np.float32).reshape(0, 2)
        
        if hasattr(self.base_dataset, 'get_raw_sample'):
            try:
                raw_sample = self.base_dataset.get_raw_sample(idx)
                fix_points = raw_sample[2] if raw_sample[2] is not None else np.array([], dtype=np.float32).reshape(0, 2)
                moving_points = raw_sample[3] if raw_sample[3] is not None else np.array([], dtype=np.float32).reshape(0, 2)
                
                img_fix = raw_sample[0]
                img_moving = raw_sample[1]
                
                if len(img_fix.shape) == 3:
                    h_fix, w_fix = img_fix.shape[:2]
                else:
                    h_fix, w_fix = img_fix.shape
                if len(img_moving.shape) == 3:
                    h_mov, w_mov = img_moving.shape[:2]
                else:
                    h_mov, w_mov = img_moving.shape
                
                scale_x_fix = self.target_size / float(w_fix)
                scale_y_fix = self.target_size / float(h_fix)
                scale_x_mov = self.target_size / float(w_mov)
                scale_y_mov = self.target_size / float(h_mov)
                
                if len(fix_points) > 0:
                    fix_points = fix_points.copy()
                    fix_points[:, 0] *= scale_x_fix
                    fix_points[:, 1] *= scale_y_fix
                if len(moving_points) > 0:
                    moving_points = moving_points.copy()
                    moving_points[:, 0] *= scale_x_mov
                    moving_points[:, 1] *= scale_y_mov
            except Exception as e:
                fix_points = np.array([], dtype=np.float32).reshape(0, 2)
                moving_points = np.array([], dtype=np.float32).reshape(0, 2)
        
        fix_points_tensor = torch.from_numpy(fix_points).float() if len(fix_points) > 0 else torch.zeros(0, 2, dtype=torch.float32)
        moving_points_tensor = torch.from_numpy(moving_points).float() if len(moving_points) > 0 else torch.zeros(0, 2, dtype=torch.float32)
        
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
            'split': self.split_name,
            'gt_pts0': fix_points_tensor,
            'gt_pts1': moving_points_tensor,
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
            # 使用硬编码绝对路径
            script_dir = Path('/data/student/Fengjunming/diffusion_registration')

            # 根据模式选择数据集
            if self.args.mode == 'cfoct':
                # CFOCT 模式: CF 为 fix, OCT 为 moving
                # 训练集: operation_pre_filtered_cfoct
                train_data_dir = script_dir / 'dataset' / 'operation_pre_filtered_cfoct'
                # 测试集: CF_OCT (使用 val 集作为验证)
                test_data_dir = script_dir / 'dataset' / 'CF_OCT'

                # 导入 CFOCT 数据集
                from dataset.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset as PreCFOCTDataset
                
                # 训练集 (使用 filtered 版本)
                train_base = PreCFOCTDataset(root_dir=str(train_data_dir), split='train', mode='oct2cf')
                self.train_dataset = RealDatasetWrapper(train_base, split_name='train', dataset_name='CFOCT')
                logger.info(f"训练集加载: {len(self.train_dataset)} 样本 (CFOCT - operation_pre_filtered_cfoct)")

                # 验证集 (使用 CF_OCT val 集) - _01=OCT, _02=CF, 所以 fix=CF, moving=OCT
                val_base = CFOCTDataset(root_dir=str(test_data_dir), split='val', mode='oct2cf')
                self.val_dataset = RealDatasetWrapper(val_base, split_name='val', dataset_name='CFOCT')
                logger.info(f"验证集加载 CFOCT val 集: {len(self.val_dataset)} 样本 (CF_OCT)")
            elif self.args.mode == 'octfa':
                # OCTFA 模式: OCT 为 fix, FA 为 moving
                data_dir = script_dir / 'dataset' / 'operation_pre_filtered_octfa'

                # 训练集
                train_base = OCTFADataset(root_dir=str(data_dir), split='train', mode='fa2oct')
                self.train_dataset = RealDatasetWrapper(train_base, split_name='train', dataset_name='OCTFA')
                logger.info(f"训练集加载: {len(self.train_dataset)} 样本 (OCTFA)")

                # 验证集（使用 val 集）- 仅使用 OCTFA 模态
                val_base = OCTFADataset(root_dir=str(data_dir), split='val', mode='fa2oct')
                self.val_dataset = RealDatasetWrapper(val_base, split_name='val', dataset_name='OCTFA')
                logger.info(f"验证集加载 OCTFA val 集: {len(self.val_dataset)} 样本")
            else:
                # CFFA 模式 (默认): CF 为 fix, FA 为 moving
                # 训练集: operation_pre_filtered_cffa
                train_data_dir = script_dir / 'dataset' / 'operation_pre_filtered_cffa'
                # 测试集: CFFA (使用 val 集作为验证)
                test_data_dir = script_dir / 'dataset' / 'CFFA'

                # 导入 CFFA 数据集
                from dataset.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset as PreCFFADataset
                
                # 训练集 (使用 filtered 版本)
                train_base = PreCFFADataset(root_dir=str(train_data_dir), split='train', mode='fa2cf')
                self.train_dataset = RealDatasetWrapper(train_base, split_name='train', dataset_name='CFFA')
                logger.info(f"训练集加载: {len(self.train_dataset)} 样本 (CFFA - operation_pre_filtered_cffa)")

                # 验证集 (使用 CFFA val 集)
                val_base = CFFADataset(root_dir=str(test_data_dir), split='val', mode='fa2cf')
                self.val_dataset = RealDatasetWrapper(val_base, split_name='val', dataset_name='CFFA')
                logger.info(f"验证集加载 CFFA val 集: {len(self.val_dataset)} 样本 (CFFA)")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, shuffle=True, collate_fn=real_batch_collate, **self.loader_params
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, shuffle=False, collate_fn=real_batch_collate, **self.loader_params
        )

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
        sp_config = config.MATCHING.get('superpoint', {})
        pretrained_path = getattr(config, 'SUPERPOINT_PRETRAINED', None)
        self.extractor = SuperPoint(sp_config, pretrained_path=pretrained_path).eval()
        for param in self.extractor.parameters():
            param.requires_grad = False
            
        # 2. 匹配器 (SuperGlue) - 可训练，加载预训练权重
        sg_config = config.MATCHING.get('superglue', {})
        superglue_pretrained_path = getattr(config, 'SUPERGLUE_PRETRAINED', None)
        self.matcher = SuperGlue(sg_config, pretrained_path=superglue_pretrained_path)
        
        self.force_viz = False
        
        # 使用统一的评估器
        self.evaluator = UnifiedEvaluator(mode='real', config=config)
        
        # 训练可视化相关
        self.train_viz_done = False
        self.train_viz_count = 0

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

    def _extract_features(self, batch):
        """提取SuperPoint特征，将list转换为batch tensor"""
        image0 = batch['image0']
        image1 = batch['image1']
        
        B = image0.shape[0]
        
        # 提取特征 (SuperPoint 返回 list 格式)
        feats0_list = self.extractor({'image': image0})
        feats1_list = self.extractor({'image': image1})
        
        # 转换为 batch tensor 格式 (SuperGlue 需要)
        keypoints0 = []
        scores0 = []
        descriptors0 = []
        keypoints1 = []
        scores1 = []
        descriptors1 = []
        
        for b in range(B):
            # image0
            keypoints0.append(feats0_list['keypoints'][b])
            scores0.append(feats0_list['scores'][b])
            descriptors0.append(feats0_list['descriptors'][b])
            
            # image1
            keypoints1.append(feats1_list['keypoints'][b])
            scores1.append(feats1_list['scores'][b])
            descriptors1.append(feats1_list['descriptors'][b])
        
        # 填充到相同长度并转换为 batch tensor
        max_kpts0 = max(k.shape[0] for k in keypoints0) if keypoints0 else 0
        max_kpts1 = max(k.shape[0] for k in keypoints1) if keypoints1 else 0
        
        # 填充 keypoints
        keypoints0_batch = torch.zeros(B, max(1, max_kpts0), 2, device=image0.device)
        keypoints1_batch = torch.zeros(B, max(1, max_kpts1), 2, device=image0.device)
        scores0_batch = torch.zeros(B, max(1, max_kpts0), device=image0.device)
        scores1_batch = torch.zeros(B, max(1, max_kpts1), device=image0.device)
        descriptors0_batch = torch.zeros(B, 256, max(1, max_kpts0), device=image0.device)
        descriptors1_batch = torch.zeros(B, 256, max(1, max_kpts1), device=image0.device)
        
        for b in range(B):
            n0 = keypoints0[b].shape[0]
            n1 = keypoints1[b].shape[0]
            
            if n0 > 0:
                keypoints0_batch[b, :n0] = keypoints0[b]
                scores0_batch[b, :n0] = scores0[b]
                descriptors0_batch[b, :, :n0] = descriptors0[b]
            
            if n1 > 0:
                keypoints1_batch[b, :n1] = keypoints1[b]
                scores1_batch[b, :n1] = scores1[b]
                descriptors1_batch[b, :, :n1] = descriptors1[b]
        
        return {
            'keypoints0': keypoints0_batch,
            'keypoints1': keypoints1_batch,
            'scores0': scores0_batch,
            'scores1': scores1_batch,
            'descriptors0': descriptors0_batch,
            'descriptors1': descriptors1_batch,
        }

    def forward(self, batch):
        # 提取特征
        with torch.no_grad():
            feats = self._extract_features(batch)
            
            # 筛选有效关键点
            keypoints0 = feats['keypoints0']
            keypoints1 = feats['keypoints1']
            scores0 = feats['scores0']
            scores1 = feats['scores1']
            descriptors0 = feats['descriptors0']
            descriptors1 = feats['descriptors1']
        
        # SuperGlue 匹配
        data = {
            'image0': batch['image0'],
            'image1': batch['image1'],
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'scores0': scores0,
            'scores1': scores1,
            'descriptors0': descriptors0,
            'descriptors1': descriptors1,
        }
        
        # 返回 SuperGlue 的输出 (包含 matches0)
        # 传递 return_scores=True 以获取用于训练的完整 Sinkhorn 输出
        outputs = self.matcher(data, return_scores=True)
        
        # 添加 keypoints 到输出以便计算损失
        outputs['keypoints0'] = keypoints0
        outputs['keypoints1'] = keypoints1
        
        return outputs

    def training_step(self, batch, batch_idx):
        """训练步骤（真实数据训练）"""
        # 可视化第一个 epoch 的前 2 个 batch
        if self.current_epoch == 0 and batch_idx < 2 and not self.train_viz_done:
            self._visualize_train_batch(batch, batch_idx)
            self.train_viz_count += 1
            if self.train_viz_count >= 2:
                self.train_viz_done = True
        
        outputs = self(batch)
        
        # 计算基于 GT 的损失
        loss = self._compute_loss(outputs, batch)
        
        # 如果损失为 None，使用简单损失
        if loss is None:
            loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _visualize_train_batch(self, batch, batch_idx):
        """可视化训练 batch 的输入数据"""
        if self.result_dir is None:
            return
            
        viz_dir = Path(self.result_dir) / 'train_viz_epoch0'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        batch_size = batch['image0'].shape[0]
        
        for i in range(batch_size):
            img0 = batch['image0'][i, 0].cpu().numpy()
            img1 = batch['image1'][i, 0].cpu().numpy()
            img1_gt = batch['image1_gt'][i, 0].cpu().numpy()
            
            img0_origin = img0.copy()
            img1_origin = img1.copy()
            
            img0 = (img0 * 255).astype(np.uint8)
            img1 = (img1 * 255).astype(np.uint8)
            img0_origin = (img0_origin * 255).astype(np.uint8)
            img1_origin = (img1_origin * 255).astype(np.uint8)
            img1_gt = (img1_gt * 255).astype(np.uint8)
            
            pair_names = batch.get('pair_names', (['unknown'] * batch_size, ['unknown'] * batch_size))
            fix_name = pair_names[0][i] if isinstance(pair_names[0], list) else pair_names[0]
            mov_name = pair_names[1][i] if isinstance(pair_names[1], list) else pair_names[1]
            
            sample_name = f"batch{batch_idx:02d}_sample{i:02d}_{Path(fix_name).stem}_vs_{Path(mov_name).stem}"
            sample_dir = viz_dir / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(sample_dir / 'fix.png'), img0)
            cv2.imwrite(str(sample_dir / 'moving.png'), img1)
            cv2.imwrite(str(sample_dir / 'fix_origin.png'), img0_origin)
            cv2.imwrite(str(sample_dir / 'moving_origin.png'), img1_origin)
            cv2.imwrite(str(sample_dir / 'moving_gt.png'), img1_gt)
            
            cb_fix_mov = create_chessboard(img1, img0, grid_size=4)
            cv2.imwrite(str(sample_dir / 'chessboard_fix_vs_moving.png'), cb_fix_mov)
            
            cb_fix_orig_mov_orig = create_chessboard(img1_origin, img0_origin, grid_size=4)
            cv2.imwrite(str(sample_dir / 'chessboard_fix_origin_vs_moving_origin.png'), cb_fix_orig_mov_orig)
            
            cb_fix_orig_mov_gt = create_chessboard(img1_gt, img0_origin, grid_size=4)
            cv2.imwrite(str(sample_dir / 'chessboard_fix_origin_vs_moving_gt.png'), cb_fix_orig_mov_gt)
            
            logger.info(f"已保存训练可视化: {sample_dir}")
        
        logger.info(f"Batch {batch_idx} 可视化完成，共 {batch_size} 个样本")

    def _compute_gt_matches(self, kpts0, kpts1, T_0to1, dist_th=3.0):
        """计算几何 Ground Truth 匹配对"""
        # kpts0, kpts1: [B, N, 2] 或 list of [N, 2]
        # T_0to1: [B, 3, 3] 从 image0 到 image1 的变换
        
        B = T_0to1.shape[0]
        device = T_0to1.device
        
        # 如果是关键点列表，转为 tensor
        if isinstance(kpts0, list):
            kpts0_tensors = []
            kpts1_tensors = []
            for b in range(B):
                kp0 = kpts0[b] if kpts0[b].shape[0] > 0 else torch.zeros(0, 2, device=device)
                kp1 = kpts1[b] if kpts1[b].shape[0] > 0 else torch.zeros(0, 2, device=device)
                kpts0_tensors.append(kp0)
                kpts1_tensors.append(kp1)
            kpts0 = torch.stack(kpts0_tensors) if all(k.shape[0] > 0 for k in kpts0_tensors) else None
            kpts1 = torch.stack(kpts1_tensors) if all(k.shape[0] > 0 for k in kpts1_tensors) else None
        
        if kpts0 is None or kpts1 is None:
            return None, None
            
        # 将关键点转换为齐次坐标并应用变换
        ones = torch.ones(B, kpts0.shape[1], 1, device=device)
        kpts0_homo = torch.cat([kpts0, ones], dim=-1)  # [B, N, 3]
        
        # 应用变换 T_0to1 (从 image0 到 image1)
        kpts0_transformed = torch.bmm(kpts0_homo, T_0to1.transpose(-2, -1))  # [B, N, 3]
        
        # 归一化齐次坐标
        kpts0_in_1 = kpts0_transformed[:, :, :2] / (kpts0_transformed[:, :, 2:3] + 1e-8)  # [B, N, 2]
        
        # 计算 kpts0_in_1 和 kpts1 之间的距离矩阵
        kpts0_exp = kpts0_in_1.unsqueeze(2)  # [B, N, 1, 2]
        kpts1_exp = kpts1.unsqueeze(1)       # [B, 1, M, 2]
        
        # 计算 L2 距离
        dist = torch.norm(kpts0_exp - kpts1_exp, dim=-1)  # [B, N, M]
        
        # 找到距离小于阈值的匹配对
        gt_matches0 = torch.argmin(dist, dim=-1)  # [B, N]
        gt_matches1 = torch.argmin(dist, dim=-2)  # [B, M]
        
        # 双向匹配
        match0_valid = torch.arange(kpts0.shape[1], device=device).unsqueeze(0).expand(B, -1)
        match1_valid = torch.arange(kpts1.shape[1], device=device).unsqueeze(0).expand(B, -1)
        
        # 检查双向匹配一致性
        valid0 = torch.gather(gt_matches1, 1, gt_matches0) == match0_valid
        valid1 = torch.gather(gt_matches0, 1, gt_matches1) == match1_valid
        
        # 返回有效的 GT 匹配
        return gt_matches0, valid0

    def _compute_loss(self, outputs, batch):
        """计算损失（SuperGlue 使用 Sinkhorn 输出）"""
        # SuperGlue 的 scores 输出是 log-space 的 Sinkhorn 结果
        # 形状: [B, M+1, N+1] (包含 dustbin)
        
        if 'scores' not in outputs:
            return None
            
        scores = outputs['scores']  # [B, M+1, N+1]
        keypoints0 = outputs.get('keypoints0', None)  # list of [N, 2]
        keypoints1 = outputs.get('keypoints1', None)  # list of [M, 2]
        
        # 获取 GT 变换矩阵
        if 'T_0to1' not in batch:
            return self._compute_simple_loss(scores)
        
        T_0to1 = batch['T_0to1']  # [B, 3, 3]
        
        # 计算 GT 匹配
        gt_matches0, valid_matches = self._compute_gt_matches(
            keypoints0, keypoints1, T_0to1, dist_th=5.0
        )
        
        if gt_matches0 is None:
            return self._compute_simple_loss(scores)
        
        # SuperGlue 交叉熵损失
        scores_trans = scores.transpose(1, 2)  # [B, N+1, M+1]
        
        B = scores.shape[0]
        device = scores.device
        
        loss = 0.0
        valid_count = 0
        
        for b in range(B):
            gt_match = gt_matches0[b]  # [N]
            valid = valid_matches[b]   # [N]
            
            if valid.sum() == 0:
                continue
            
            max_idx = scores_trans.shape[2] - 1
            gt_match_clamped = gt_match.clamp(0, max_idx)
            
            # 提取正样本分数（匹配位置）
            pos_scores = scores_trans[b][torch.arange(len(gt_match), device=device), gt_match_clamped]
            
            # 交叉熵损失
            log_probs = pos_scores - torch.logsumexp(scores_trans[b, :len(gt_match)], dim=-1)
            
            # 只对有效匹配计算损失
            valid_log_probs = log_probs[valid]
            if len(valid_log_probs) > 0:
                loss = loss - valid_log_probs.mean()
                valid_count += 1
        
        if valid_count > 0:
            loss = loss / valid_count
        else:
            loss = self._compute_simple_loss(scores)
        
        return loss
    
    def _compute_simple_loss(self, scores):
        """计算简单的损失：最大化匹配分数"""
        # scores: [B, M+1, N+1]
        # 鼓励更高的匹配分数
        
        # 使用 dustbin 的分数作为负样本
        dustbin_scores = scores[:, -1, :]  # [B, N+1]
        keypoint_scores = scores[:, :-1, :]  # [B, M, N+1]
        
        # 简单的损失：鼓励正样本分数高于 dustbin
        loss = -torch.mean(keypoint_scores) + 0.1 * torch.mean(dustbin_scores)
        
        return loss

    def _extract_and_match(self, batch):
        """提取特征并匹配（用于验证）"""
        # 提取特征
        feats = self._extract_features(batch)
        
        keypoints0 = feats['keypoints0']
        keypoints1 = feats['keypoints1']
        scores0 = feats['scores0']
        scores1 = feats['scores1']
        descriptors0 = feats['descriptors0']
        descriptors1 = feats['descriptors1']
        
        # SuperGlue 匹配
        data = {
            'image0': batch['image0'],
            'image1': batch['image1'],
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'scores0': scores0,
            'scores1': scores1,
            'descriptors0': descriptors0,
            'descriptors1': descriptors1,
        }
        
        outputs = self.matcher(data)
        
        # 保留原始关键点用于后续处理
        outputs['keypoints0'] = keypoints0
        outputs['keypoints1'] = keypoints1
        
        return outputs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """验证步骤（使用统一的评估器）"""
        outputs = self._extract_and_match(batch)
        
        # 将关键点添加到 batch 中（因为 evaluator 从 batch 读取 keypoints0/1）
        batch['keypoints0'] = outputs['keypoints0']
        batch['keypoints1'] = outputs['keypoints1']
        
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
        self.result_dir = Path(f"/data/student/Fengjunming/diffusion_registration/SuperGluePretrainedNetwork/results/superglue_{args.mode}/{args.name}")
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
                    
                    if sample_split in ('test', 'val'):
                        self._process_batch_sample(trainer, pl_module, batch, outputs, target_dir, i, batch_idx, sample_split)
                        visualized_count += 1
                        
                        if visualized_count >= max_visualize:
                            break
                
                if visualized_count >= max_visualize:
                    break
        
        logger.info(f"已可视化 {visualized_count} 个验证/测试集样本")
        pl_module.force_viz = False

    def _process_batch_sample(self, trainer, pl_module, batch, outputs, epoch_dir, sample_idx, batch_idx, split):
        """处理并可视化单个样本"""
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
        
        img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        
        if 'keypoints0' in outputs and 'keypoints1' in outputs:
            kpts0_np = outputs['keypoints0'][sample_idx].cpu().numpy()
            kpts1_np = outputs['keypoints1'][sample_idx].cpu().numpy()
            
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
                    fig = plt.figure(figsize=(12, 6))
                    from lightglue import viz2d
                    viz2d.plot_images([img0, img1])
                    if len(m_indices_0) > 0 and len(m_indices_1) > 0:
                        viz2d.plot_matches(kpts0_np[m_indices_0], kpts1_np[m_indices_1], color='lime', lw=0.5)
                    plt.savefig(str(save_path / "matches.png"), bbox_inches='tight', dpi=100)
                    plt.close('all')
                except Exception as e:
                    logger.warning(f"绘制匹配图失败: {e}")
                    try:
                        plt.close('all')
                    except:
                        pass
        
        cv2.imwrite(str(save_path / "fix_with_kpts.png"), img0_color)
        cv2.imwrite(str(save_path / "moving_with_kpts.png"), img1_color)
        
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
    parser = argparse.ArgumentParser(description="SuperGlue Real-Data Training")
    parser.add_argument('--name', '-n', type=str, default='superglue_baseline', help='训练名称')
    parser.add_argument('--mode', type=str, default='cffa', choices=['cffa', 'cfoct', 'octfa'], help='数据集模式: cffa, cfoct 或 octfa')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--start_point', type=str, default=None, help='从检查点恢复')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--gpus', type=str, default='1')
    parser.add_argument('--superpoint_pretrained', type=str, default=None,
                        help='SuperPoint 预训练权重路径 (默认从 models/weights/ 加载，如果不存在会自动下载)')
    parser.add_argument('--superglue_pretrained', type=str, default=None,
                        help='SuperGlue 预训练权重路径 (默认从 models/weights/ 加载 indoor 权重)')
    parser.add_argument('--patience', type=int, default=10, help='早停和学习率调度的 patience 值')
    return parser.parse_args()

def main():
    args = parse_args()

    config = get_default_config()
    config.TRAINER.SEED = 66
    config.TRAINER.PATIENCE = args.patience  # 使用传入的 patience 值
    pl.seed_everything(config.TRAINER.SEED)

    # 设置 SuperPoint 预训练权重路径
    if args.superpoint_pretrained:
        config.SUPERPOINT_PRETRAINED = args.superpoint_pretrained
    else:
        # 默认路径：如果不存在会自动下载
        default_sp_path = Path(__file__).parent.parent / 'models' / 'weights' / 'superpoint_v1.pth'
        config.SUPERPOINT_PRETRAINED = str(default_sp_path)
        logger.info(f"SuperPoint 预训练权重路径: {config.SUPERPOINT_PRETRAINED}")
    
    # 设置 SuperGlue 预训练权重路径
    if args.superglue_pretrained:
        config.SUPERGLUE_PRETRAINED = args.superglue_pretrained
    else:
        # 默认路径：加载 indoor 权重
        default_sg_path = Path(__file__).parent.parent / 'models' / 'weights' / 'superglue_indoor.pth'
        config.SUPERGLUE_PRETRAINED = str(default_sg_path)
        logger.info(f"SuperGlue 预训练权重路径: {config.SUPERGLUE_PRETRAINED}")

    result_dir = Path(f"/data/student/Fengjunming/diffusion_registration/SuperGluePretrainedNetwork/results/superglue_{args.mode}/{args.name}")
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
        patience=args.patience,
        min_delta=0.0001,
        strict=False
    )
    
    logger.info(f"早停配置: monitor=combined_auc, start_epoch=0, patience={args.patience}, min_delta=0.0001")

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
    
    logger.info(f"开始 SuperGlue 真实数据训练: {args.name}")
    logger.info("模型: SuperGlue + SuperPoint")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()
