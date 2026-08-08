import jax
import jax.numpy as jnp
from utils.param_config import get_base_params
base = get_base_params()
k_smooth = 20.0  # tanh光滑系数
# 1 风切变扰动 Ws(r, theta) 三阶泰勒展开
def wind_shear(r, theta, alpha, H=base["H"], R=base["R"]):
    term1 = alpha * (r / H) * jnp.cos(theta)
    term2 = alpha * (alpha - 1) / 2 * (r / H) ** 2 * jnp.cos(theta) ** 2
    term3 = alpha * (alpha - 1) * (alpha - 2) / 6 * (r / H) ** 3 * jnp.cos(theta) ** 3
    return term1 + term2 + term3
# 2 塔影光滑替代阶跃区间 [pi/2, 3pi/2]
def tower_shadow_mask(theta):
    mask = 1.0 / (1 + jnp.exp(-k_smooth * (theta - jnp.pi/2))) * \
           1.0 / (1 + jnp.exp(k_smooth * (theta - 3 * jnp.pi/2)))
    return mask
# 3 塔影等效风速扰动 vtower
def vtower(r, theta, Vh, alpha):
    a = base["a_tower"]
    x = base["x_blade_tower"]
    R = base["R"]
    H = base["H"]
    m = 1 + alpha * (alpha - 1) * R**2 / (8 * H**2)
    mask = tower_shadow_mask(theta)
    numer = m * a**2 * (r**2 * jnp.sin(theta)**2 - x**2)
    denom = (r**2 * jnp.sin(theta)**2 + x**2) ** 2
    return mask * Vh * numer / denom
# 4 单叶片等效风速积分简化（三叶片求和）
def v_eq_single_blade(theta_b, Vh, alpha):
    r_arr = jnp.linspace(0, base["R"], 30)
    # 向量化遍历r_arr每个径向位置
    ws = jnp.vectorize(lambda r: wind_shear(r, theta_b, alpha))(r_arr)
    vt = jnp.vectorize(lambda r: vtower(r, theta_b, Vh, alpha))(r_arr)
    # 修复：把 r 替换为径向数组 r_arr
    integrand = r_arr * (1 + ws + vt)
    integral = jnp.trapz(integrand, r_arr)
    return 2 / base["R"]**2 * integral * Vh
# 5 三叶片等效风速【修改：取消平均，单叶片保留偏航特征】
@jax.jit
def calc_veq(p):
    Vh, alpha, theta_yaw, beta = p
    veq = v_eq_single_blade(theta_yaw, Vh, alpha)
    return veq
# 6 可微Cp(λ, β) 风能利用系数
def calc_cp(lam, beta):
    c1,c2,c3,c4,c5,c6 = base["c1"],base["c2"],base["c3"],base["c4"],base["c5"],base["c6"]
    inv_lam = 1 / (lam + 0.08 * beta) - 0.035 / (beta**3 + 1)
    cp = c1 * (c2 / inv_lam - c3 * beta - c4) * jnp.exp(-c5 / inv_lam) + c6 * lam
    return jnp.clip(cp, -0.1, 0.59)
# 7 气动转矩 Tm(p)
@jax.jit
def calc_Tm(p):
    Vh, alpha, theta_yaw, beta = p
    veq = calc_veq(p)
    omega_r = 1.2  # 基准转子转速 rad/s 稳态点
    lam = omega_r * base["R"] / veq
    cp = calc_cp(lam, beta)
    Pm = 0.5 * base["rho"] * jnp.pi * base["R"]**2 * veq**3 * cp
    Tm = Pm / omega_r
    return Tm