import torch
import cv2
import numpy as np
from collections import OrderedDict
from loguru import logger
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous
import sys
import os

# 【终极方案】创建一个同时输出到控制台和文件的日志函数
_log_file = None
_verbose_mode = False  # 全局标志：是否输出详细日志（默认关闭，仅在验证时开启）

def set_metrics_verbose(verbose):
    """设置是否输出详细的指标日志（用于区分训练/验证阶段）"""
    global _verbose_mode
    _verbose_mode = verbose

def _dual_log(level, message):
    """同时输出到 loguru 和文件"""
    global _log_file, _verbose_mode
    
    # 如果不在 verbose 模式，只输出 WARNING 和 ERROR
    if not _verbose_mode and level == "INFO":
        return
    
    # 输出到 loguru
    if level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    
    # 同时直接写入文件（绕过 loguru）
    if _log_file is None:
        # 尝试从环境变量获取日志文件路径
        log_path = os.environ.get('LOFTR_LOG_FILE', None)
        if log_path:
            try:
                _log_file = open(log_path, 'a', buffering=1)
            except:
                pass
    
    if _log_file:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log_file.write(f"{timestamp} | {level: <8} | {message}\n")
        _log_file.flush()


# --- METRICS ---

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d


def compute_symmetrical_epipolar_errors(data):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    dataset_name = data['dataset_name'][0].lower()
    # 【修复】必须包含所有医学眼底数据集名称
    multimodal_datasets = ['multimodal', 'realdataset', 'cffa', 'cfoct', 'octfa', 'cfocta']
    if dataset_name in multimodal_datasets:
        return compute_homography_reprojection_errors(data)
    
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = Tx @ data['T_0to1'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']

    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(pts0[mask], pts1[mask], E_mat[bs], data['K0'][bs], data['K1'][bs]))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})


def compute_homography_reprojection_errors(data):
    """
    计算基于单应矩阵的重投影误差 (针对 MultiModal 数据集)
    Update:
        data (dict):{"epi_errs": [M]}
    """
    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']
    H_0to1 = data['T_0to1'] # [N, 3, 3]

    epi_errs = []
    for bs in range(H_0to1.size(0)):
        mask = m_bids == bs
        if mask.sum() == 0:
            continue
        
        # 将 pts0 投影到 image1
        p0 = pts0[mask]
        p0_h = torch.cat([p0, torch.ones_like(p0[:, :1])], dim=-1)
        p0_warped_h = (H_0to1[bs] @ p0_h.t()).t()
        p0_warped = p0_warped_h[:, :2] / (p0_warped_h[:, 2:] + 1e-7)
        
        # 计算 L2 距离作为误差
        err = torch.norm(p0_warped - pts1[mask], dim=-1)
        epi_errs.append(err)
        
    if len(epi_errs) == 0:
        data.update({'epi_errs': torch.tensor([], device=pts0.device)})
    else:
        data.update({'epi_errs': torch.cat(epi_errs, dim=0)})


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def compute_pose_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    dataset_name = data['dataset_name'][0].lower()
    # 【修复】必须包含所有医学眼底数据集名称
    multimodal_datasets = ['multimodal', 'realdataset', 'cffa', 'cfoct', 'octfa', 'cfocta']
    if dataset_name in multimodal_datasets:
        return compute_homography_errors(data, config)
    
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({
        'R_errs': [],
        't_errs': [],
        'inliers': [],
        'failed_mask': [],
        'inaccurate_mask': [],
    })

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        ret = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf)

        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(bool))
        else:
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            data['R_errs'].append(R_err)
            data['t_errs'].append(t_err)
            data['inliers'].append(inliers)


def spatial_binning(pts0, pts1, img_size, grid_size=4, top_n=20, conf=None):
    """
    【方案 B 改进】空间均匀化 (Spatial Binning)
    将图像划分为 grid_size x grid_size 的网格，每个网格内最多保留 Top-N 个匹配点。
    """
    h, w = img_size
    cell_h = h / grid_size
    cell_w = w / grid_size
    
    selected_indices = []
    
    # 模拟网格
    grid = [[] for _ in range(grid_size * grid_size)]
    
    for i, pt in enumerate(pts0):
        gx = min(int(pt[0] / cell_w), grid_size - 1)
        gy = min(int(pt[1] / cell_h), grid_size - 1)
        grid[gy * grid_size + gx].append(i)
        
    for cell_indices in grid:
        if len(cell_indices) == 0:
            continue
        
        if conf is not None:
            # 按置信度排序
            cell_indices = sorted(cell_indices, key=lambda idx: conf[idx], reverse=True)
            
        selected_indices.extend(cell_indices[:top_n])
        
    return np.array(selected_indices)


def compute_homography_errors(data, config):
    """
    计算单应矩阵估计误差 (针对 MultiModal 数据集)
    新增：返回 mae, mee 用于判断 inaccurate
    """
    data.update({'R_errs': [], 't_errs': [], 'inliers': [], 'H_est': [], 'mae': [], 'mee': [],
                 'mse_list': [], 'mace_list': [], 'failed_mask': [], 'inaccurate_mask': []})
    
    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    H_gt = data['T_0to1'].cpu().numpy()
    mconf = data.get('mconf')
    if mconf is not None:
        mconf = mconf.cpu().numpy()
    
    for bs in range(H_gt.shape[0]):
        mask = m_bids == bs
        num_matches = np.sum(mask)
        
        if num_matches < 4:
            _dual_log("WARNING", f"⚠️ Batch {bs}: 匹配点数不足 ({num_matches} < 4)")
            # 【修复】Failed 样本：R_errs=0.0, t_errs=1e6（用于AUC计算）
            data['R_errs'].append(0.0)
            data['t_errs'].append(1e6)
            data['inliers'].append(np.array([]).astype(bool))
            data['H_est'].append(np.eye(3))
            data['mae'].append(np.inf)
            data['mee'].append(np.inf)
            data['mse_list'].append(np.inf)
            data['mace_list'].append(np.inf)
            data['failed_mask'].append(True)
            data['inaccurate_mask'].append(False)
            continue
            
        # 估计单应矩阵 (对应 plan.md 第四阶段: 几何估计)
        # 【方案 B 改进】进行空间均匀化 (Spatial Binning)
        img_size = data['image0'].shape[2:]
        pts0_batch = pts0[mask]
        pts1_batch = pts1[mask]
        mconf_batch = mconf[mask] if mconf is not None else None
        
        bin_indices = spatial_binning(pts0_batch, pts1_batch, img_size, grid_size=4, top_n=20, conf=mconf_batch)
        
        _dual_log("INFO", f"🔍 Batch {bs}: 总匹配点={num_matches}, Spatial Binning后={len(bin_indices)}")
        
        if len(bin_indices) >= 4:
            pts0_ransac = pts0_batch[bin_indices]
            pts1_ransac = pts1_batch[bin_indices]
            H_est, inliers = cv2.findHomography(pts0_ransac, pts1_ransac, cv2.RANSAC, config.TRAINER.RANSAC_PIXEL_THR)
        else:
            H_est, inliers = cv2.findHomography(pts0_batch, pts1_batch, cv2.RANSAC, config.TRAINER.RANSAC_PIXEL_THR)
        
        if H_est is None:
            # 【调试】RANSAC 失败，记录原因
            _dual_log("WARNING", f"⚠️ Batch {bs}: RANSAC 返回 None (匹配点数: {len(bin_indices) if len(bin_indices) >= 4 else num_matches})")
            # 【修复】Failed 样本：R_errs=0.0, t_errs=1e6（用于AUC计算）
            data['R_errs'].append(0.0)
            data['t_errs'].append(1e6)
            data['inliers'].append(np.array([]).astype(bool))
            data['H_est'].append(np.eye(3))
            data['mae'].append(np.inf)
            data['mee'].append(np.inf)
            data['mse_list'].append(np.inf)
            data['mace_list'].append(np.inf)
            data['failed_mask'].append(True)
            data['inaccurate_mask'].append(False)
        else:
            # 【调试】检查 inliers 数量和矩阵状态
            num_inliers = np.sum(inliers.ravel() > 0) if inliers is not None else 0
            # 【修复】添加 inliers_rate 检查，使用全部匹配点作为分母
            denom_inliers = len(pts0_batch)
            inliers_rate = num_inliers / max(1, int(denom_inliers))
            is_identity = np.allclose(H_est, np.eye(3), atol=1e-3)

            _dual_log("INFO", f"✅ Batch {bs}: RANSAC 成功, inliers={num_inliers}/{denom_inliers}, inliers_rate={inliers_rate:.2e}, H_est是否单位矩阵={is_identity}")

            # 【修复】先检查 inliers_rate 是否小于 1e-6
            is_failed = False
            if inliers_rate < 1e-6:
                _dual_log("WARNING", f"⚠️ Batch {bs}: inliers_rate < 1e-6，判定为 Failed")
                is_failed = True

            if is_identity:
                _dual_log("WARNING", f"⚠️ Batch {bs}: H_est 接近单位矩阵! 匹配失败")
                is_failed = True

            if is_failed:
                # 【修复】Failed 样本：R_errs=0.0, t_errs=1e6（用于AUC计算）
                data['R_errs'].append(0.0)
                data['t_errs'].append(1e6)
                data['inliers'].append(np.array([]).astype(bool))
                data['H_est'].append(np.eye(3))
                data['mae'].append(np.inf)
                data['mee'].append(np.inf)
                data['mse_list'].append(np.inf)
                data['mace_list'].append(np.inf)
                data['failed_mask'].append(True)
                data['inaccurate_mask'].append(False)
            else:
                if num_inliers < 30:
                    _dual_log("WARNING", f"⚠️ Batch {bs}: Inliers 数量较少 ({num_inliers}), 可能导致配准质量差")

                # 对于眼底图像配准，我们将 R_errs 设为 0
                data['R_errs'].append(0.0)

                # 重要修复：计算 Corner Error (估计 H 与 真值 H 之间的偏差)
                h, w = data['image0'].shape[2:]
                corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                corners_h = np.concatenate([corners, np.ones((4, 1))], axis=-1)

                # 使用真值 H 投影得到 GT 坐标
                corners_gt_h = (H_gt[bs] @ corners_h.T).T
                corners_gt = corners_gt_h[:, :2] / (corners_gt_h[:, 2:] + 1e-7)

                # 使用估计 H 投影得到预测坐标
                corners_est_h = (H_est @ corners_h.T).T
                corners_est = corners_est_h[:, :2] / (corners_est_h[:, 2:] + 1e-7)

                # 计算平均角点误差 (MACE)
                err = np.mean(np.linalg.norm(corners_est - corners_gt, axis=-1))

                data['t_errs'].append(err)
                data['inliers'].append(inliers.ravel() > 0)
                data['H_est'].append(H_est)

                # 计算点对点误差，用于判断 inaccurate 和 MSE
                pts0_batch = pts0[mask]
                pts1_batch = pts1[mask]
                mae, mee, dis = compute_pointwise_errors(pts0_batch, pts1_batch, H_est)
                data['mae'].append(mae)
                data['mee'].append(mee)

                # 计算 MSE（点对点投影误差的均方值）
                mse_val = float(np.mean(dis ** 2)) if len(dis) > 0 else np.inf

                # 根据 inaccurate 条件决定是否记录 MSE/MACE（t_errs 已在上方按真实误差记录，参与 AUC）
                if is_inaccurate(mae, mee):
                    data['mse_list'].append(np.inf)
                    data['mace_list'].append(np.inf)
                    data['failed_mask'].append(False)
                    data['inaccurate_mask'].append(True)
                else:
                    data['mse_list'].append(mse_val)
                    data['mace_list'].append(float(err))
                    data['failed_mask'].append(False)
                    data['inaccurate_mask'].append(False)


def compute_pointwise_errors(pts0, pts1, H_est):
    """
    计算点对点的投影误差（用于判断 inaccurate）

    Args:
        pts0: 源图像匹配点 [N, 2]
        pts1: 目标图像匹配点 [N, 2]
        H_est: 估计的单应矩阵 [3, 3]

    Returns:
        mae: 最大误差 (Maximum Absolute Error)
        mee: 中位误差 (Median Error)
        dis: 所有点的误差数组 [N]
    """
    if pts0 is None or len(pts0) == 0:
        return np.inf, np.inf, np.array([])

    pts0_h = np.concatenate([pts0, np.ones((pts0.shape[0], 1))], axis=-1)
    pts1_pred_h = (H_est @ pts0_h.T).T
    pts1_pred = pts1_pred_h[:, :2] / (pts1_pred_h[:, 2:] + 1e-7)

    dis = np.sqrt(np.sum((pts1 - pts1_pred) ** 2, axis=1))
    mae = np.max(dis)
    mee = np.median(dis)

    return mae, mee, dis


# --- INACCURATE 判断阈值 ---
# 与 test_on_CrossModality.py 保持一致
INACCURATE_MAE_THRESHOLD = 50.0  # 最大误差 > 50 像素视为 inaccurate
INACCURATE_MEE_THRESHOLD = 20.0  # 中位误差 > 20 像素视为 inaccurate


def is_inaccurate(mae, mee):
    """
    判断样本是否 inaccurate（不准确但未失败）

    Args:
        mae: 最大误差
        mee: 中位误差

    Returns:
        bool: True 表示 inaccurate
    """
    return mae > INACCURATE_MAE_THRESHOLD or mee > INACCURATE_MEE_THRESHOLD

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    # 使用传入的 thresholds（例如 [10, 20, 40]），而不是在函数内部重写
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def compute_auc_rop(errors, limit=25):
    """
    计算 ROP 风格的 mAUC (按照 test_on_CrossModality.py 的方式)
    Args:
        errors (list): 误差列表
        limit (int): 阈值上限，默认25像素
    Returns:
        dict: {'mAUC': float}
    """
    errors = np.array(errors)
    gs_error = np.zeros(limit + 1)
    accum_s = 0
    for i in range(1, limit + 1):
        gs_error[i] = np.sum(errors < i) * 100 / len(errors)
        accum_s = accum_s + gs_error[i]
    
    auc_s = accum_s / (limit * 100)
    return {'mAUC': auc_s}


def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs


def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc（对 MultiModal 场景，误差单位为像素，这里统一使用 [5, 10, 20] 像素阈值）
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)

    return {**aucs, **precs}
