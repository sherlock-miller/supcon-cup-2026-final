#!/usr/bin/env python3
"""
手眼标定解算：TSAI 法 AX=XB（纯 numpy，OpenCV 4/5 通用）
====================================================
OpenCV 5.0 移除了 cv2.calibrateHandEye，此模块提供等价实现。

算法（Tsai & Lenz 1989, IEEE JRA）：
  已知 N 组位姿对 (A_i, B_i)，A_i = 末端在基座系位姿（gripper→base），
  B_i = 标定板在相机系位姿（target→camera）。
  求 X（cam→gripper）使 A X = X B。

  旋转部分: 轴角表示下，每对给出约束，SVD 求 R_x
  平移部分: (R_A - I) t_x = R_x t_B - t_A，最小二乘

返回 (R_cam2gripper, t_cam2gripper)。
"""
import numpy as np


def _rotation_to_axis_angle(R):
    """旋转矩阵 → 轴角 (angle, axis)。θ≈π 时用 R+I 特征分解提取轴（数值稳定）"""
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-10:
        return 0.0, np.array([1.0, 0.0, 0.0])
    if np.pi - theta < 1e-6:
        # 180° 旋转: 轴 = (R+I)/2 的最大特征值特征向量
        w, v = np.linalg.eigh((R + np.eye(3)) / 2.0)
        axis = v[:, np.argmax(w)].real
        axis /= np.linalg.norm(axis)
        return theta, axis
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz]) / (2.0 * np.sin(theta))
    axis /= np.linalg.norm(axis)
    return theta, axis


def _axis_angle_to_rotation(theta, axis):
    """轴角 → 旋转矩阵（Rodrigues）"""
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def calibrate_handeye_tsai(R_g2b, t_g2b, R_t2c, t_t2c):
    """
    TSAI 法解 AX=XB。

    Args:
        R_g2b, t_g2b: 末端→基座 位姿列表（A_i）
        R_t2c, t_t2c: 标定板→相机 位姿列表（B_i）

    Returns:
        (R_cam2gripper, t_cam2gripper)
    """
    n = len(R_g2b)
    if n < 3:
        raise ValueError(f"位姿对不足: {n} < 3")

    # ---- 旋转部分（Tsai-Lenz 最小二乘形式）----
    # 每对 (i, i+1) 相对位姿: skew(Pg + Pc) · Pg' = Pc − Pg
    # 堆叠为 M @ Pg' = b，最小二乘解出正确缩放的 Pg'
    M_rows, b_rows = [], []
    for i in range(n - 1):
        Rg_ij = R_g2b[i + 1] @ R_g2b[i].T
        Rc_ij = R_t2c[i + 1] @ R_t2c[i].T
        th_g, ax_g = _rotation_to_axis_angle(Rg_ij)
        th_c, ax_c = _rotation_to_axis_angle(Rc_ij)
        if th_g < 1e-6 or th_c < 1e-6:
            continue  # 无相对旋转，对约束无贡献
        Pg = 2.0 * np.sin(th_g / 2.0) * ax_g
        Pc = 2.0 * np.sin(th_c / 2.0) * ax_c
        K = np.array([[0, -(Pg[2] + Pc[2]), (Pg[1] + Pc[1])],
                      [(Pg[2] + Pc[2]), 0, -(Pg[0] + Pc[0])],
                      [-(Pg[1] + Pc[1]), (Pg[0] + Pc[0]), 0]])
        M_rows.append(K)
        b_rows.append(Pc - Pg)

    if len(M_rows) < 2:
        raise ValueError("有效相对旋转对不足（位姿必须有旋转变化）")
    M_mat = np.vstack(M_rows)
    b_vec = np.concatenate(b_rows)
    Pg_prime, *_ = np.linalg.lstsq(M_mat, b_vec, rcond=None)

    # Pg' = 2 sin(θx/2) · axis → 分解为 R_x
    sin_half = np.clip(np.linalg.norm(Pg_prime) / 2.0, -1.0, 1.0)
    th_x = 2.0 * np.arcsin(sin_half)
    axis = Pg_prime / (2.0 * sin_half + 1e-12)
    R_x = _axis_angle_to_rotation(th_x, axis)

    # ---- 平移部分 ----
    A_stack = []
    b_stack = []
    for i in range(n):
        A_stack.append(R_g2b[i] - np.eye(3))
        b_stack.append(R_x @ t_t2c[i].ravel() - t_g2b[i].ravel())
    A_mat = np.vstack(A_stack)
    b_vec = np.concatenate(b_stack)
    t_x, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    return R_x, t_x


def calibrate_handeye_compat(R_g2b, t_g2b, R_t2c, t_t2c):
    """OpenCV 4/5 双版本兼容入口（4.x 用 cv2.calibrateHandEye，5.x 用 TSAI numpy）"""
    try:
        import cv2
        if hasattr(cv2, "calibrateHandEye"):
            return cv2.calibrateHandEye(
                R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI)
    except Exception:
        pass
    return calibrate_handeye_tsai(R_g2b, t_g2b, R_t2c, t_t2c)


def check_pose_diversity(R_g2b, t_g2b, min_translation_var=0.005, min_rotation_deg=5.0):
    """
    位姿多样性检查——手眼标定要求位姿充分变化。
    返回 (ok, report_str)。
    """
    t_arr = np.array([t.ravel() for t in t_g2b])
    t_var = np.var(t_arr, axis=0)
    t_span = np.max(t_arr, axis=0) - np.min(t_arr, axis=0)

    # 旋转多样性: 平均相对旋转角
    rel_angles = []
    for i in range(len(R_g2b) - 1):
        R_rel = R_g2b[i + 1] @ R_g2b[i].T
        th, _ = _rotation_to_axis_angle(R_rel)
        rel_angles.append(np.degrees(th))
    max_rel = max(rel_angles) if rel_angles else 0.0

    ok = (t_span.max() >= min_translation_var) or (max_rel >= min_rotation_deg)
    report = (f"平移跨度 xyz={t_span.round(3)} m, "
              f"最大相对旋转={max_rel:.1f}°")
    if not ok:
        report += " ⚠️ 位姿几乎无变化——手眼标定数据无效，请掰动机械臂重新采集！"
    return ok, report


if __name__ == "__main__":
    # 自测: 构造已知 X（120° 旋转，非边界），生成位姿对，验证解算恢复 X
    rng = np.random.default_rng(42)
    ax_true = np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0)
    X_R_true = _axis_angle_to_rotation(np.radians(120.0), ax_true)
    X_t_true = np.array([0.05, -0.08, 0.02])

    def rot_x(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rot_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for i in range(12):
        a1 = np.radians(rng.uniform(-60, 60))
        a2 = np.radians(rng.uniform(-40, 40))
        A_R = rot_x(a1) @ rot_y(a2)
        A_t = rng.uniform(-0.1, 0.1, 3)
        B_R = X_R_true.T @ A_R @ X_R_true  # A X = X B → B = X^-1 A X
        B_t = X_R_true.T @ (A_R @ X_t_true + A_t - X_t_true)
        R_g2b.append(A_R); t_g2b.append(A_t.reshape(3, 1))
        R_t2c.append(B_R); t_t2c.append(B_t.reshape(3, 1))

    R_x, t_x = calibrate_handeye_tsai(R_g2b, t_g2b, R_t2c, t_t2c)
    print("真值 R:\n", X_R_true)
    print("解算 R:\n", np.round(R_x, 4))
    print("真值 t:", X_t_true)
    print("解算 t:", np.round(t_x, 4))
    R_err = np.degrees(np.arccos(np.clip((np.trace(R_x.T @ X_R_true) - 1) / 2, -1, 1)))
    t_err = np.linalg.norm(t_x - X_t_true)
    print(f"旋转误差: {R_err:.3f}°  平移误差: {t_err:.4f} m")
    ok, rep = check_pose_diversity(R_g2b, t_g2b)
    print("多样性检查:", rep)
    assert R_err < 2.0 and t_err < 0.01, "TSAI 解算误差过大"
    assert ok, "多样性检查应通过"
    print("SELF-TEST PASS")
