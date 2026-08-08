import jax
import jax.numpy as jnp
import diffrax
from utils.param_config import get_base_params
from model.aero_model import calc_Tm

# 加载风机机械固定参数
base = get_base_params()
Jr, Jg = base["Jr"], base["Jg"]
Dr, Dg, Dsh, Ksh = base["Dr"], base["Dg"], base["Dsh"], base["Ksh"]


# 双质量块轴系ODE右端函数 dx/dt = f(t, state, args)
# state = [wr:转子转速, wg:发电机转速, th_sh:轴系扭转角]
def ode_rhs(t, state, Tm):
    wr, wg, th_sh = state
    # 转子动力学
    d_wr = (Tm - Ksh * th_sh - Dsh * (wr - wg) - Dr * wr) / Jr
    # 发电机动力学
    d_wg = (Ksh * th_sh + Dsh * (wr - wg) - Dg * wg) / Jg
    # 扭转角微分 = 转速差
    d_thsh = wr - wg
    return jnp.array([d_wr, d_wg, d_thsh])


# 求解稳态发电机转速wg，兼容所有diffrax版本、可jax.jit
@jax.jit
def solve_omega_g(p):
    # 输入气动参数计算气动转矩
    Tm = calc_Tm(p)

    # 仿真时间区间
    t_start = 0.0
    t_end = 10.0
    # 初始状态 [转子转速,发电机转速,初始扭转角0]
    x0 = jnp.array([1.2, 1.18, 0.0])

    # 微分方程标准组件（MultiTerm兼容新旧diffrax，不会报无Terms属性）
    ode_term = diffrax.ODETerm(ode_rhs)
    terms = diffrax.MultiTerm(ode_term)

    # 5阶龙格库塔自适应求解器
    solver = diffrax.Tsit5()
    # 仅保存终点时刻结果，标准数组传参，修复SaveAt参数报错
    save_spec = diffrax.SaveAt(ts=jnp.array([t_end]))

    # 求解ODE，补齐全部必填参数，无缺失参数报错
    sol = diffrax.diffeqsolve(
        terms=terms,
        solver=solver,
        t0=t_start,
        t1=t_end,
        dt0=0.01,  # 初始积分步长，必填
        y0=x0,
        args=Tm,
        saveat=save_spec
    )
    # sol.ys 形状 (采样点数, 3)，取终点、第二个分量 = wg
    wg = sol.ys[0, 1]
    return wg