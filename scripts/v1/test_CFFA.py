"""
统一的测试/验证模块
提供 gen 和 cffa 两种模式的前向推理和指标计算接口
供训练脚本和独立测试脚本调用，确保指标计算的一致性
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
import random

# 添加父目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scripts.v1.metrics import (
    compute_homography_errors, 
    set_metrics_verbose,
    error_auc,
    compute_auc_rop
)


# ============ CFFA 数据集包装器 (与 train_onReal.py 对齐) ============
class CFFADatasetWrapper(torch.utils.data.Dataset):
    """CFFA 数据集包装器 - 将 CFFADataset 转换为 SuperGlue 需要的格式"""
    def __init__(self, base_dataset, split_name='unknown'):
        self.base_dataset = base_dataset
        self.split_name = split_name

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        # 将 [-1, 1] 转换为 [0, 1]
        moving_original_tensor = (moving_original_tensor + 1) / 2
        moving_gt_tensor = (moving_gt_tensor + 1) / 2

        # 转换为灰度图
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

        # T_0to1 是从 moving 到 fix 的变换，这里需要存储逆变换
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
            'dataset_name': 'CFFA',
            'split': self.split_name
        }


class CFFADataModule(pl.LightningDataModule):
    """CFFA 数据模块 - 加载训练集+测试集用于测试"""
    def __init__(self, args, config, data_dir=None):
        super().__init__()
        self.args = args
        self.config = config
        # 默认使用 data/CFFA 目录
        if data_dir is None:
            script_dir = Path(__file__).parent.parent.parent
            self.data_dir = script_dir / 'data' / 'CFFA'
        else:
            self.data_dir = Path(data_dir)
        self.loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': True
        }

    def setup(self, stage=None):
        # 导入 CFFA 数据集
        from data.CFFA.cffa_dataset import CFFADataset

        # 加载训练集
        train_base = CFFADataset(root_dir=str(self.data_dir), split='train', mode='cf2fa')
        self.train_dataset = CFFADatasetWrapper(train_base, split_name='train')
        logger.info(f"训练集加载: {len(self.train_dataset)} 样本")

        # 加载验证/测试集
        val_base = CFFADataset(root_dir=str(self.data_dir), split='val', mode='cf2fa')
        self.val_dataset = CFFADatasetWrapper(val_base, split_name='test')
        logger.info(f"测试集加载: {len(self.val_dataset)} 样本")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=False, **self.loader_params)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle=False, **self.loader_params)


def filter_valid_area(img1, img2):
    """筛选有效区域：只保留两张图片都不为纯黑像素的部分"""
    assert img1.shape[:2] == img2.shape[:2], "两张图片的尺寸必须一致"
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
    """计算四个角点的平均重投影误差（MACE）"""
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


class UnifiedEvaluator:
    """
    统一的评估器
    支持 gen 和 cffa 两种模式
    累积所有 batch 的误差后统一计算 AUC
    """
    def __init__(self, mode='gen', config=None):
        """
        Args:
            mode: 'gen' 或 'cffa'
            config: 配置对象，包含 TRAINER.RANSAC_PIXEL_THR 等参数
        """
        self.mode = mode
        self.config = config
        self.reset()
    
    def reset(self):
        """重置累积的指标"""
        self.all_errors = []  # 累积所有样本的误差用于计算 AUC
        self.all_mses = []    # 累积所有样本的 MSE
        self.all_maces = []   # 累积所有样本的 MACE
        self.total_samples = 0
        self.failed_samples = 0
        self.inaccurate_samples = 0  # 新增：inaccurate 样本数（不准确但未失败）
    
    def evaluate_batch(self, batch, outputs, pl_module):
        """
        评估一个 batch
        
        Args:
            batch: 数据 batch，包含 image0, image1, image1_gt, T_0to1, keypoints0, keypoints1 等
            outputs: 模型输出，包含 matches0 等
            pl_module: PyTorch Lightning 模块（用于获取 config）
            
        Returns:
            dict: 包含 H_est, mses, maces, metrics_batch 等信息
        """
        matches0 = outputs['matches0']
        kpts0 = batch['keypoints0']
        kpts1 = batch['keypoints1']
        
        B = kpts0.shape[0]
        H_ests = []
        batch_mses = []
        batch_maces = []
        
        # 构建用于 metrics.py 的数据格式
        mkpts0_f_list = []
        mkpts1_f_list = []
        m_bids_list = []
        
        # 为每张图计算单应矩阵
        for b in range(B):
            self.total_samples += 1
            
            m0 = matches0[b]
            valid = m0 > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[valid]
            
            pts0 = kpts0[b][m_indices_0].cpu().numpy()
            pts1 = kpts1[b][m_indices_1].cpu().numpy()
            
            # 保存匹配点（用于 metrics.py 计算 AUC）
            if len(pts0) > 0:
                mkpts0_f_list.append(torch.from_numpy(pts0).float())
                mkpts1_f_list.append(torch.from_numpy(pts1).float())
                m_bids_list.append(torch.full((len(pts0),), b, dtype=torch.long))
            
            # 计算单应矩阵
            if len(pts0) >= 4:
                try:
                    ransac_thr = self.config.TRAINER.RANSAC_PIXEL_THR if self.config else 3.0
                    H, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransac_thr)
                    if H is None:
                        H = np.eye(3)
                except:
                    H = np.eye(3)
            else:
                H = np.eye(3)
            
            # 判断是否匹配失败（与 test_on_CrossModality.py 对齐）
            is_match_failed = False
            if H is None or np.allclose(H, np.eye(3), atol=1e-3) or len(pts0) < 4:
                is_match_failed = True
                H = np.eye(3)
            
            H_ests.append(H)
            
            # 计算 MSE、avg_dist、MACE（只在匹配成功时）
            # 统一在这里判断是否 failed（与 test_on_CrossModality.py 对齐：inliers < 1e-6）
            # 先计算 inliers_rate
            try:
                _, inliers = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransac_thr)
                if inliers is not None:
                    inliers_count = np.sum(inliers.ravel() > 0)
                    inliers_rate = inliers_count / len(pts0) if len(pts0) > 0 else 0
                else:
                    inliers_rate = 0
            except:
                inliers_rate = 0
            
            # 判断是否 failed（与 test_on_CrossModality.py 对齐：inliers < 1e-6）
            # 或者原本就是无效的 H（单位矩阵或检测失败）
            if inliers_rate < 1e-6 or is_match_failed:
                self.failed_samples += 1
                # failed 样本不计算 MSE，用 inf 填充
                batch_mses.append(float('inf'))
                batch_maces.append(float('inf'))
                self.all_mses.append(float('inf'))
                self.all_maces.append(float('inf'))
                # 记录一个很大的 error 用于 AUC
                self.all_errors.append(1e6)
            else:
                # 计算 MSE（特征点坐标 MSE，与 test_on_CrossModality.py 的 cal_MSE 对齐）
                # 使用 cv2.perspectiveTransform 将 pts0 用 H 变换，然后计算 MSE
                pts0_homo = pts0.reshape(-1, 1, 2).astype(np.float32)
                pts1_pred = cv2.perspectiveTransform(pts0_homo, H).reshape(-1, 2)
                
                # 计算 MSE（特征点坐标 MSE）
                mse = np.mean((pts1 - pts1_pred) ** 2)
                
                # 计算 avg_dist（匹配点的平均重投影误差，与 test_on_CrossModality.py 对齐）
                dis = (pts1 - pts1_pred) ** 2
                dis = np.sqrt(dis[:, 0] + dis[:, 1])
                avg_dist = dis.mean()
                
                # 计算 MAE 和 MEE（用于判断 inaccurate）
                mae = dis.max()
                mee = np.median(dis)
                
                # 计入 AUC（所有成功样本都计入，包括 inaccurate）
                self.all_errors.append(avg_dist)
                
                # 判断 inaccurate（与 test_on_CrossModality.py 对齐：mae > 50 或 mee > 20）
                is_inaccurate = mae > 50.0 or mee > 20.0
                if is_inaccurate:
                    self.inaccurate_samples += 1
                    # 与 test_on_CrossModality.py 对齐：inaccurate 样本不记录 MSE/MACE
                    batch_mses.append(float('inf'))
                    batch_maces.append(float('inf'))
                    self.all_mses.append(float('inf'))
                    self.all_maces.append(float('inf'))
                else:
                    # 计算 MACE（角点误差）
                    T_gt = batch['T_0to1'][b].cpu().numpy()
                    # compute_corner_error 需要图像尺寸；从 batch 中读取即可（B,1,H,W）
                    h, w = batch['image0'].shape[-2], batch['image0'].shape[-1]
                    mace = compute_corner_error(H, T_gt, h, w)
                    
                    # MSE 只记录非 inaccurate 的样本（与 test_on_CrossModality.py 对齐）
                    batch_mses.append(mse)
                    batch_maces.append(mace)
                    self.all_mses.append(mse)
                    self.all_maces.append(mace)
        
        # 构建 metrics.py 需要的 batch 格式
        metrics_batch = {
            'mkpts0_f': torch.cat(mkpts0_f_list, dim=0) if mkpts0_f_list else torch.empty(0, 2),
            'mkpts1_f': torch.cat(mkpts1_f_list, dim=0) if mkpts1_f_list else torch.empty(0, 2),
            'm_bids': torch.cat(m_bids_list, dim=0) if m_bids_list else torch.empty(0, dtype=torch.long),
            'T_0to1': batch['T_0to1'],
            'image0': batch['image0'],
            'dataset_name': batch['dataset_name']
        }
        
        # 使用 metrics.py 计算指标（会填充 t_errs, mae, mee）
        set_metrics_verbose(False)  # 训练时不输出详细日志
        compute_homography_errors(metrics_batch, self.config if self.config else pl_module.config)

        # 注意：all_errors 已在上面的循环中累积（使用 avg_dist，与 test_on_CrossModality.py 对齐）
        # metrics.py 计算的 t_errs（角点误差）不再使用

        # 注意：inaccurate 判断已在上面的循环中完成（与 test_on_CrossModality.py 对齐）
        # 不再重复计算
        
        return {
            'H_est': H_ests,
            'mses': batch_mses,
            'maces': batch_maces,
            'metrics_batch': metrics_batch,
            'matches0': matches0,  # 添加 matches0 用于可视化
            'kpts0': kpts0,        # 添加 kpts0 用于可视化
            'kpts1': kpts1         # 添加 kpts1 用于可视化
        }
    
    def compute_epoch_metrics(self):
        """
        计算整个 epoch 的聚合指标
        
        Returns:
            dict: 包含 auc@5, auc@10, auc@20, mAUC, combined_auc, mse, mace, match_failure_rate 等
        """
        metrics = {}
        
        # 计算 AUC（使用所有样本统一计算）
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
        
        # 计算 MSE 和 MACE（只在匹配成功的样本上，排除 inf 值）
        valid_mses = [m for m in self.all_mses if not np.isinf(m)]
        valid_maces = [m for m in self.all_maces if not np.isinf(m)]
        metrics['mse'] = sum(valid_mses) / len(valid_mses) if valid_mses else 0.0
        metrics['mace'] = sum(valid_maces) / len(valid_maces) if valid_maces else 0.0
        metrics['inverse_mace'] = 1.0 / (1.0 + metrics['mace']) if metrics['mace'] > 0 else 1.0
        
        # 计算匹配失败率
        metrics['match_failure_rate'] = self.failed_samples / self.total_samples if self.total_samples > 0 else 0.0
        metrics['total_samples'] = self.total_samples
        metrics['failed_samples'] = self.failed_samples
        metrics['success_samples'] = self.total_samples - self.failed_samples

        # 计算 inaccurate 样本统计（与 test_on_CrossModality.py 对齐）
        # inaccurate = 匹配成功但 mae > 50 或 mee > 20
        metrics['inaccurate_samples'] = self.inaccurate_samples
        metrics['acceptable_samples'] = self.total_samples - self.failed_samples - self.inaccurate_samples
        metrics['inaccurate_rate'] = self.inaccurate_samples / self.total_samples if self.total_samples > 0 else 0.0
        metrics['acceptable_rate'] = metrics['acceptable_samples'] / self.total_samples if self.total_samples > 0 else 0.0

        return metrics


def run_evaluation(pl_module, dataloader, mode='gen', verbose=True, save_visualizations=False, output_dir=None):
    """
    运行完整的评估流程
    
    Args:
        pl_module: PyTorch Lightning 模块
        dataloader: 数据加载器
        mode: 'gen' 或 'cffa'
        verbose: 是否输出详细日志
        save_visualizations: 是否保存可视化结果
        output_dir: 可视化结果保存目录
        
    Returns:
        dict: 评估指标
    """
    evaluator = UnifiedEvaluator(mode=mode, config=pl_module.config)
    
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
        logger.info(f"评估完成: {metrics}")
    
    return metrics


def _visualize_batch(batch, outputs, output_dir, batch_idx):
    """可视化一个batch的结果"""
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    batch_size = batch['image0'].shape[0]
    H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
    
    for sample_idx in range(batch_size):
        H_est = H_ests[sample_idx]
        
        # 与 test_on_CrossModality.py 对齐：不使用防爆锁
        
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
        if pair_names:
            sample_name = f"batch{batch_idx:04d}_sample{sample_idx:02d}_{Path(pair_names[0][sample_idx]).stem}_vs_{Path(pair_names[1][sample_idx]).stem}"
        else:
            sample_name = f"batch{batch_idx:04d}_sample{sample_idx:02d}"
        
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
                
                # 绘制匹配连线 (替换 lightglue.viz2d)
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


# ==========================================
# 主函数
# ==========================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SuperGlue 统一测试脚本")
    parser.add_argument('--mode', type=str, required=True, choices=['gen', 'real'],
                        help='模型类型: gen (train_onGen_vessels.py训练) 或 real (train_onReal.py训练)')
    parser.add_argument('--name', type=str, required=True,
                        help='模型名称（用于定位检查点）')
    parser.add_argument('--test_name', type=str, required=True,
                        help='测试名称（结果保存在 results/superglue_[gen|real]/[name]/[test_name] 下）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='检查点路径（默认使用 best_checkpoint/model.ckpt）')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--gpus', type=str, default='0', help='GPU设备ID')
    parser.add_argument('--img_size', type=int, default=512, help='图像大小')
    parser.add_argument('--test_split', type=str, default='both', choices=['train', 'test', 'both'],
                        help='测试数据集选择: train (仅训练集), test (仅测试集), both (训练集+测试集)')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 导入必要的模块（根据mode动态导入）
    if args.mode == 'gen':
        # 导入生成数据训练的模型
        from scripts.v1.train_onGen_vessels import (
            PL_SuperGlue_Gen,
            get_default_config
        )
        pl_class = PL_SuperGlue_Gen
        mode_dir = 'superglue_gen'
    else:  # real
        # 导入真实数据训练的模型
        from scripts.v1.train_onReal import (
            PL_SuperGlue_Real,
            get_default_config
        )
        pl_class = PL_SuperGlue_Real
        mode_dir = 'superglue_cffa'
    
    # 始终使用 CFFA 数据集进行测试
    data_module_class = CFFADataModule
    
    # 获取配置
    config = get_default_config()
    pl.seed_everything(config.TRAINER.SEED)
    
    # 确定checkpoint路径
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path(f"results/{mode_dir}/{args.name}/best_checkpoint/model.ckpt")
    
    if not ckpt_path.exists():
        logger.error(f"检查点不存在: {ckpt_path}")
        logger.info(f"请确保训练模型存在，或使用 --checkpoint 指定有效的检查点路径")
        return
    
    logger.info(f"加载检查点: {ckpt_path}")
    
    # 设置输出目录
    output_dir = Path(f"results/{mode_dir}/{args.name}/{args.test_name}")
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
    
    config.TRAINER.WORLD_SIZE = max(_n_gpus, 1)
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    
    logger.info(f"测试模式: {args.mode}")
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
    data_module = data_module_class(args, config)
    data_module.setup('fit')  # 使用 'fit' 来加载验证集

    # 测试数据集：使用 --test_split 参数指定：'train', 'test', 或 'both'
    # 始终在 CFFA 数据集上测试
    test_split = getattr(args, 'test_split', 'both')

    if test_split == 'both':
        # 合并训练集和测试集进行测试
        logger.info("测试模式: CFFA 训练集 + 测试集 合并测试")
        train_dataloader = data_module.train_dataloader()
        test_dataloader = data_module.val_dataloader()

        # 分别测试并合并结果
        set_metrics_verbose(True)

        # 测试训练集
        logger.info("=" * 50)
        logger.info("开始测试 CFFA 训练集")
        logger.info("=" * 50)
        metrics_train = run_evaluation(
            model,
            train_dataloader,
            mode=args.mode,
            verbose=True,
            save_visualizations=False,
            output_dir=output_dir / 'train_set'
        )

        # 测试测试集
        logger.info("=" * 50)
        logger.info("开始测试 CFFA 测试集")
        logger.info("=" * 50)
        metrics_test = run_evaluation(
            model,
            test_dataloader,
            mode=args.mode,
            verbose=True,
            save_visualizations=True,
            output_dir=output_dir / 'test_set'
        )

        # 合并两个数据集的结果
        combined_metrics = {
            'train_samples': metrics_train['total_samples'],
            'test_samples': metrics_test['total_samples'],
            'total_samples': metrics_train['total_samples'] + metrics_test['total_samples'],
            'train_success': metrics_train['success_samples'],
            'test_success': metrics_test['success_samples'],
            'success_samples': metrics_train['success_samples'] + metrics_test['success_samples'],
            'train_failed': metrics_train['failed_samples'],
            'test_failed': metrics_test['failed_samples'],
            'failed_samples': metrics_train['failed_samples'] + metrics_test['failed_samples'],
            'train_inaccurate': metrics_train['inaccurate_samples'],
            'test_inaccurate': metrics_test['inaccurate_samples'],
            'inaccurate_samples': metrics_train['inaccurate_samples'] + metrics_test['inaccurate_samples'],
            'train_acceptable': metrics_train['acceptable_samples'],
            'test_acceptable': metrics_test['acceptable_samples'],
            'acceptable_samples': metrics_train['acceptable_samples'] + metrics_test['acceptable_samples'],
        }
        combined_metrics['match_failure_rate'] = combined_metrics['failed_samples'] / combined_metrics['total_samples'] if combined_metrics['total_samples'] > 0 else 0
        combined_metrics['inaccurate_rate'] = combined_metrics['inaccurate_samples'] / combined_metrics['total_samples'] if combined_metrics['total_samples'] > 0 else 0
        combined_metrics['acceptable_rate'] = combined_metrics['acceptable_samples'] / combined_metrics['total_samples'] if combined_metrics['total_samples'] > 0 else 0
        combined_metrics['mse'] = (metrics_train['mse'] * metrics_train['total_samples'] + metrics_test['mse'] * metrics_test['total_samples']) / combined_metrics['total_samples'] if combined_metrics['total_samples'] > 0 else 0
        combined_metrics['mace'] = (metrics_train['mace'] * metrics_train['total_samples'] + metrics_test['mace'] * metrics_test['total_samples']) / combined_metrics['total_samples'] if combined_metrics['total_samples'] > 0 else 0
        combined_metrics['inverse_mace'] = 1.0 / (1.0 + combined_metrics['mace']) if combined_metrics['mace'] > 0 else 1.0
        combined_metrics['auc@5'] = (metrics_train['auc@5'] + metrics_test['auc@5']) / 2
        combined_metrics['auc@10'] = (metrics_train['auc@10'] + metrics_test['auc@10']) / 2
        combined_metrics['auc@20'] = (metrics_train['auc@20'] + metrics_test['auc@20']) / 2
        combined_metrics['mAUC'] = (metrics_train['mAUC'] + metrics_test['mAUC']) / 2
        combined_metrics['combined_auc'] = (combined_metrics['auc@5'] + combined_metrics['auc@10'] + combined_metrics['auc@20']) / 3.0

        metrics = combined_metrics
    elif test_split == 'train':
        # 仅测试训练集
        logger.info("测试模式: CFFA 仅训练集")
        test_dataloader = data_module.train_dataloader()
        set_metrics_verbose(True)
        metrics = run_evaluation(
            model,
            test_dataloader,
            mode=args.mode,
            verbose=True,
            save_visualizations=True,
            output_dir=output_dir
        )
    else:
            # 默认仅测试测试集
            logger.info("测试模式: CFFA 仅测试集")
            test_dataloader = data_module.val_dataloader()
            set_metrics_verbose(True)
            metrics = run_evaluation(
                model,
                test_dataloader,
                mode=args.mode,
                verbose=True,
                save_visualizations=True,
                output_dir=output_dir
            )
    
    # 保存测试总结
    summary_path = output_dir / "test_summary.txt"
    with open(summary_path, "w") as f:
        f.write("测试总结\n")
        f.write("=" * 50 + "\n")
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
    
    logger.info(f"测试总结已保存到: {summary_path}")
    logger.info(f"测试完成! 结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
