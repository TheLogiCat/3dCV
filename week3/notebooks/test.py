#!/usr/bin/env python
"""
演示：简单的全局对齐（Sim(3)）前后点云对比。
依赖: numpy, matplotlib
可选: open3d (若要 3D 交互查看)
运行: python align_pointcloud_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']  # 依次回退
rcParams['axes.unicode_minus'] = False  # 负号正常显示

# ========== 1. 生成参考点云 ========== #
def make_reference_pointcloud(n_points=5000, seed=42):
    """
    生成一个带结构的点云（椭球 + 少许随机噪声）
    """
    rng = np.random.default_rng(seed)
    # 采样在单位球内
    pts = rng.normal(size=(n_points, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    # 拉伸成椭球
    scale_axes = np.array([2.0, 1.0, 0.5])
    pts = pts * scale_axes
    # 加一些随机散点
    noise = 0.05 * rng.normal(size=(n_points, 3))
    pts += noise
    return pts

# ========== 2. 施加随机 Sim(3) 变换得到“第二视图”点云 ========== #
def random_sim3_transform(seed=7):
    rng = np.random.default_rng(seed)
    # 随机旋转（用随机四元数）
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1-2*(x*x+z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1-2*(x*x+y*y)]
    ])
    s = rng.uniform(0.7, 1.4)    # 尺度
    t = rng.uniform(-0.5, 0.5, size=3)  # 平移
    return s, R, t

def apply_sim3(pts, s, R, t):
    return s * (pts @ R.T) + t

# ========== 3. Umeyama 算法（估计 Sim(3)） ========== #
def umeyama_alignment(src, dst, with_scale=True):
    """
    给定 src 点 (N×3) 通过相似变换对齐到 dst (N×3).
    返回: s, R, t 使得 dst ≈ s*R*src + t
    """
    assert src.shape == dst.shape
    n = src.shape[0]
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_c = src - mean_src
    dst_c = dst - mean_dst

    # 协方差
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt

    if with_scale:
        var_src = np.sum(src_c**2) / n
        s = (D * np.diag(S)).sum() / var_src
    else:
        s = 1.0

    t = mean_dst - s * R @ mean_src
    return s, R, t

# ========== 4. 可视化 ========== #
def scatter_compare(ref, other_before, other_after, sample=3000):
    """
    2D 投影对比（简单使用前两个主成分或直接 XY 平面）
    """
    # 随机采样减少渲染开销
    rng = np.random.default_rng(0)
    idx_ref = rng.choice(ref.shape[0], min(sample, ref.shape[0]), replace=False)
    idx_o1  = rng.choice(other_before.shape[0], min(sample, other_before.shape[0]), replace=False)
    idx_o2  = rng.choice(other_after.shape[0], min(sample, other_after.shape[0]), replace=False)

    ref_s = ref[idx_ref]
    before_s = other_before[idx_o1]
    after_s = other_after[idx_o2]

    # 为了对比直观，可以用 PCA 前两维（这里简单用 XY）
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    axs[0].scatter(ref_s[:,0], ref_s[:,1], s=5, c='C0', alpha=0.6, label='Reference')
    axs[0].scatter(before_s[:,0], before_s[:,1], s=5, c='C3', alpha=0.6, label='Before Align')
    axs[0].set_title("对齐前 (XY 投影)")
    axs[0].axis('equal')
    axs[0].legend()

    axs[1].scatter(ref_s[:,0], ref_s[:,1], s=5, c='C0', alpha=0.6, label='Reference')
    axs[1].scatter(after_s[:,0], after_s[:,1], s=5, c='C2', alpha=0.6, label='After Align')
    axs[1].set_title("对齐后 (XY 投影)")
    axs[1].axis('equal')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def main():
    # 生成参考点云
    ref = make_reference_pointcloud(n_points=8000)

    # 生成第二视图点云（未知变换 + 噪声）
    s_gt, R_gt, t_gt = random_sim3_transform()
    view2 = apply_sim3(ref, s_gt, R_gt, t_gt)

    # 加少许噪声模拟预测误差
    noise = 0.01 * np.random.default_rng(1).normal(size=view2.shape)
    view2_noisy = view2 + noise

    # 估计对齐（将 view2_noisy 对齐回 ref）
    s_est, R_est, t_est = umeyama_alignment(view2_noisy, ref, with_scale=True)
    aligned = apply_sim3(view2_noisy, s_est, R_est, t_est)

    # 误差评估
    rmse_before = np.sqrt(np.mean((view2_noisy - ref)**2))
    rmse_after  = np.sqrt(np.mean((aligned - ref)**2))

    print("真实变换: scale=%.4f" % s_gt)
    print("估计变换: scale=%.4f" % s_est)
    print("对齐前 RMSE: %.6f" % rmse_before)
    print("对齐后 RMSE: %.6f" % rmse_after)

    scatter_compare(ref, view2_noisy, aligned)

if __name__ == "__main__":
    main()