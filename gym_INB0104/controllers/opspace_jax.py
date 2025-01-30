# opspace_jax.py
from typing import Optional, Tuple, Union
import jax
import jax.numpy as jnp
from jax import jit

def pseudo_inverse(M, damped=False, lambda_=0.2):
    if not damped:
        return jnp.linalg.pinv(M)
    U, sing_vals, V = jnp.linalg.svd(M, full_matrices=False)
    S_inv = sing_vals / (sing_vals**2 + lambda_**2)
    return (V.T * S_inv) @ U.T

def pd_control(x, x_des, dx, kp_kv, max_pos_error=0.05):
    x_err = jnp.clip(x - x_des, -max_pos_error, max_pos_error)
    dx_err = dx
    return -kp_kv[:, 0] * x_err - kp_kv[:, 1] * dx_err

def quat_diff_active(source_quat, target_quat):
    source_inv = jnp.array([source_quat[0], -source_quat[1], -source_quat[2], -source_quat[3]])
    w1, x1, y1, z1 = target_quat
    w2, x2, y2, z2 = source_inv
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    diff_quat = jnp.array([w, x, y, z])
    norm = jnp.linalg.norm(diff_quat)
    return diff_quat / jnp.where(norm == 0, 1.0, norm)

def quat_to_axisangle(quat):
    angle = 2 * jnp.arccos(jnp.clip(quat[0], -1.0, 1.0))
    sin_half = jnp.sin(angle / 2)
    safe_sin = jnp.where(sin_half < 1e-7, 1.0, sin_half)
    axis = quat[1:] / safe_sin
    axis = jnp.where(jnp.abs(angle) < 1e-7, jnp.zeros_like(axis), axis)
    return axis * angle

def pd_control_orientation(quat, quat_des, w, kp_kv, max_ori_error=0.05):
    quat_err = quat_diff_active(quat_des, quat)
    ori_err = jnp.clip(quat_to_axisangle(quat_err), -max_ori_error, max_ori_error)
    w_err = w
    return -kp_kv[:, 0] * ori_err - kp_kv[:, 1] * w_err

def saturate_torque_rate(tau_calculated, tau_prev, delta_tau_max):
    return tau_prev + jnp.clip(tau_calculated - tau_prev, -delta_tau_max, delta_tau_max)

def mat_to_quat(mat):
    trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
    cond = jnp.array([trace > 0, 
                     (mat[0,0] > mat[1,1]) & (mat[0,0] > mat[2,2]),
                     (mat[1,1] > mat[2,2])], dtype=bool)
    
    def case0(_):
        s = jnp.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (mat[2,1] - mat[1,2]) / s
        y = (mat[0,2] - mat[2,0]) / s
        z = (mat[1,0] - mat[0,1]) / s
        return jnp.array([w, x, y, z])
    
    def case1(_):
        s = jnp.sqrt(1.0 + mat[0,0] - mat[1,1] - mat[2,2]) * 2.0
        w = (mat[2,1] - mat[1,2]) / s
        x = 0.25 * s
        y = (mat[0,1] + mat[1,0]) / s
        z = (mat[0,2] + mat[2,0]) / s
        return jnp.array([w, x, y, z])
    
    def case2(_):
        s = jnp.sqrt(1.0 + mat[1,1] - mat[0,0] - mat[2,2]) * 2.0
        w = (mat[0,2] - mat[2,0]) / s
        x = (mat[0,1] + mat[1,0]) / s
        y = 0.25 * s
        z = (mat[1,2] + mat[2,1]) / s
        return jnp.array([w, x, y, z])
    
    def case3(_):
        s = jnp.sqrt(1.0 + mat[2,2] - mat[0,0] - mat[1,1]) * 2.0
        w = (mat[1,0] - mat[0,1]) / s
        x = (mat[0,2] + mat[2,0]) / s
        y = (mat[1,2] + mat[2,1]) / s
        z = 0.25 * s
        return jnp.array([w, x, y, z])
    
    quat = jax.lax.cond(
        cond[0], case0,
        lambda _: jax.lax.cond(
            cond[1], case1,
            lambda _: jax.lax.cond(
                cond[2], case2,
                case3, None
            ), None
        ), None
    )
    return quat / jnp.linalg.norm(quat)

@jit
def opspace_jax(
    site_xpos,
    site_xmat,
    q,
    dq,
    J_v,
    J_w,
    qfrc_bias,
    qpos_actuator,
    pos_des,
    ori_des,
    joint_des,
    pos_gains=jnp.array([1500.0, 1500.0, 1500.0]),
    ori_gains=jnp.array([200.0, 200.0, 200.0]),
    joint_upper_limits=jnp.array([2.8, 1.7, 2.8, -0.08, 2.8, 3.74, 2.8]),
    joint_lower_limits=jnp.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.010, -2.8]),
    translational_damping=89.0,
    rotational_damping=7.0,
    nullspace_stiffness=0.2,
    joint1_nullspace_stiffness=100.0,
    max_pos_error=0.01,
    max_ori_error=0.01,
    delta_tau_max=0.5,
    gravity_comp=True,
    damped=False,
    lambda_=0.2,
):
    x_des = pos_des
    quat_des = ori_des

    kp_pos = pos_gains
    kd_pos = translational_damping * jnp.ones_like(kp_pos)
    kp_kv_pos = jnp.stack([kp_pos, kd_pos], axis=-1)

    kp_ori = ori_gains
    kd_ori = rotational_damping * jnp.ones_like(kp_ori)
    kp_kv_ori = jnp.stack([kp_ori, kd_ori], axis=-1)

    J = jnp.vstack((J_v, J_w))
    dx = J_v @ dq
    ddx = pd_control(site_xpos, x_des, dx, kp_kv_pos, max_pos_error)

    quat = mat_to_quat(site_xmat.reshape(3, 3))
    quat = jnp.where(quat @ quat_des < 0.0, -quat, quat)
    w = J_w @ dq
    dw = pd_control_orientation(quat, quat_des, w, kp_kv_ori, max_ori_error)

    F = jnp.concatenate([ddx, dw])
    tau_task = J.T @ F

    q_error = joint_des - q
    q_error = q_error.at[0].multiply(joint1_nullspace_stiffness)
    dq_error = dq
    dq_error = dq_error.at[0].multiply(2 * jnp.sqrt(joint1_nullspace_stiffness))
    tau_nullspace = nullspace_stiffness * q_error - 2 * jnp.sqrt(nullspace_stiffness) * dq_error

    if damped:
        jacobian_transpose_pinv = pseudo_inverse(J.T, damped=True, lambda_=lambda_)
        tau_nullspace = (jnp.eye(len(q)) - J.T @ jacobian_transpose_pinv) @ tau_nullspace

    tau = tau_task + tau_nullspace

    mask_upper = (q >= joint_upper_limits) & (tau > 0)
    mask_lower = (q <= joint_lower_limits) & (tau < 0)
    tau = jnp.where(mask_upper | mask_lower, 0.0, tau)

    if gravity_comp:
        tau += qfrc_bias

    return saturate_torque_rate(tau, qpos_actuator, delta_tau_max)