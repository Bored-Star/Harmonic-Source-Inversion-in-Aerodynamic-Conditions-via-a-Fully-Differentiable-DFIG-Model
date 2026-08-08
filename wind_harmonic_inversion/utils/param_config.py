import jax.numpy as jnp

# DFIG风机全局固定参数（论文仿真参数表）
def get_base_params():
    params = {
        # 气动几何
        "rho": 1.225,       # 空气密度 kg/m³
        "R": 40.0,          # 叶片半径 m
        "H": 80.0,          # 轮毂高度 m
        "a_tower": 0.85,    # 塔筒半径 m
        "x_blade_tower": 2.9, # 叶片-塔筒轴向距离 m
        # Cp曲线系数
        "c1": 0.5176, "c2": 116, "c3": 0.4, "c4": 5, "c5": 21, "c6": 0.0068,
        # 机械双质量块
        "Jr": 3.2e6,        # 风轮惯量 kg·m²
        "Jg": 1.2e4,        # 发电机惯量
        "Dr": 800, "Dg": 100,
        "Dsh": 3200, "Ksh": 1.2e6,
        # 电气DFIG参数
        "Ls": 0.08, "Lr": 0.075, "Lm": 2.9,
        "Rs": 0.012, "Rr": 0.014,
        "f1": 50, "w1": 2 * jnp.pi * 50,
        # 仿真时间
        "t_start": 0.0, "t_end": 30.0, "t_step": 0.01
    }
    return params

# 待反演源头参数边界 p=[Vh, alpha, theta_yaw, beta]
def get_param_bounds():
    bounds = jnp.array([
        [3.0, 25.0],    # Vh 轮毂风速
        [0.05, 0.5],    # alpha 风切变指数
        [-jnp.pi/6, jnp.pi/6], # 偏航角 rad ±30°
        [0, jnp.pi/3]   # 桨距角 0~60°
    ])
    return bounds