import jax
import jax.numpy as jnp
from model.aero_model import calc_Tm
from model.mech_model import solve_omega_g
from model.elec_model import calc_harmonic

# 全局正向模型：输入源头参数p，输出观测 [fsh, Ish]
@jax.jit
def forward_model(p):
    wg = solve_omega_g(p)
    fsh, Ish = calc_harmonic(wg)
    y = jnp.array([fsh, Ish])
    return y

# 批量正向（生成合成数据集用）
batch_forward = jax.vmap(forward_model)