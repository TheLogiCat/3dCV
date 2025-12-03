import numpy as np
from scipy.spatial.transform import Rotation as R

def solve_transformation():
    # ================= 1. 数据准备 =================
    # 注意：书中惯用四元数顺序为 [w, x, y, z] (实部在前)
    # 而 scipy 的 Rotation 库默认顺序为 [x, y, z, w] (实部在后)
    
    # 小萝卜一号 (Robot 1)
    q1_raw = np.array([0.35, 0.2, 0.3, 0.1]) # [w, x, y, z]
    t1 = np.array([0.3, 0.1, 0.1])
    
    # 小萝卜二号 (Robot 2)
    q2_raw = np.array([-0.5, 0.4, -0.1, 0.2]) # [w, x, y, z]
    t2 = np.array([-0.1, 0.5, 0.3])
    
    # 观测到的点 (在 R1 坐标系下)
    p_r1 = np.array([0.5, 0, 0.2])

    # ================= 2. 辅助函数 =================
    def get_transform_matrix(q_raw, t):
        """
        根据原始四元数和平移向量构建 4x4 变换矩阵 T_rw (World -> Robot)
        """
        # 1. 归一化四元数 (题目给的数据模长不为1，必须归一化)
        q_norm = q_raw / np.linalg.norm(q_raw)
        
        # 2. 转换为 scipy 格式 [x, y, z, w]
        q_scipy = [q_norm[1], q_norm[2], q_norm[3], q_norm[0]]
        
        # 3. 计算旋转矩阵 R
        r_matrix = R.from_quat(q_scipy).as_matrix()
        
        # 4. 构建 4x4 变换矩阵 T
        T = np.eye(4)
        T[:3, :3] = r_matrix
        T[:3, 3] = t
        return T

    # ================= 3. 计算过程 =================
    
    # 计算变换矩阵
    # T1_w: 世界 -> R1
    T_r1_w = get_transform_matrix(q1_raw, t1)
    
    # T2_w: 世界 -> R2
    T_r2_w = get_transform_matrix(q2_raw, t2)
    
    # 将点转换为齐次坐标 [x, y, z, 1]
    p_r1_homo = np.append(p_r1, 1)
    
    # 第一步：求世界坐标系下的点 P_w
    # 因为 P_r1 = T_r1_w * P_w，所以 P_w = T_r1_w的逆 * P_r1
    T_w_r1 = np.linalg.inv(T_r1_w)
    p_w_homo = np.dot(T_w_r1, p_r1_homo)
    
    # 第二步：求小萝卜二号坐标系下的点 P_r2
    # P_r2 = T_r2_w * P_w
    p_r2_homo = np.dot(T_r2_w, p_w_homo)
    
    # 取出前三维作为结果
    p_r2 = p_r2_homo[:3]

    # ================= 4. 输出结果 =================
    print(f"小萝卜一号坐标系下的点 P_r1: {p_r1}")
    print(f"计算出的世界坐标系下的点 P_w : {p_w_homo[:3]}")
    print(f"小萝卜二号坐标系下的点 P_r2: {p_r2}")
    
    return p_r2

if __name__ == "__main__":
    solve_transformation()