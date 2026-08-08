import jax
import jax.numpy as jnp
from utils.param_config import get_base_params

base = get_base_params()
# 正确拆分4个电气参数
Ls = base["Ls"]
Lr = base["Lr"]
Lm = base["Lm"]
Rs = base["Rs"]
Rr = base["Rr"]
w1 = base["w1"]

# 给定wg输出主导次同步间谐波频率fsh、幅值Ish
@jax.jit
def calc_harmonic(wg):
    fr = wg
    slip = (w1 - fr) / w1
    # 主导次同步分量 h=1 典型工况
    h = 1
    fsh = jnp.abs(h * base["f1"] - fr / (2 * jnp.pi))
    wsh = 2 * jnp.pi * fsh
    # 等效导纳公式
    denom = (Rs + 1j * wsh * Ls) * (Rr / slip + 1j * wsh * Lr) + (1j * wsh * Lm)**2
    Yrs = 1j * wsh * Lm / denom
    Yrsdq = (1j * wsh * Lm * (Rs + 1j * wsh * Ls)) / denom
    Urh = 0.02  # 扰动电压幅值标幺
    Ish = Urh * jnp.abs(Yrs + Yrsdq)
    return fsh, Ish