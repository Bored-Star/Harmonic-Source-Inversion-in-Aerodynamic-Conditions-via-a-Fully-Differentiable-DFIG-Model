import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from algorithm.loss_func import loss_fn
from algorithm.grad_solver import grad_fd, grad_auto
from utils.param_config import get_param_bounds

bounds = get_param_bounds()
lb, ub = bounds[:, 0], bounds[:, 1]

def lbfgs_optim(p_init, y_obs, use_auto_grad=True):
    def cost(p):
        p_jax = jnp.array(p)
        loss_val = loss_fn(p_jax, y_obs, p_init)
        return float(loss_val)

    def grad_func(p):
        if use_auto_grad:
            p_jax = jnp.array(p)
            grad_jax = grad_auto(p_jax, y_obs, p_init)
            return jnp.asarray(grad_jax)
        else:
            return grad_fd(p, y_obs, p_init)
    # 增加最大迭代次数，降低梯度收敛阈值，充分搜索极小值
    res = minimize(
        fun=cost,
        x0=p_init,
        jac=grad_func,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 2000,
            "gtol": 1e-10
        }
    )
    return res.x, res.fun

# 两轮独立随机采样合并，样本总量500，提升覆盖度
def bayes_search(y_obs, sample_num=250):
    key = jax.random.PRNGKey(42)
    sample1 = jax.random.uniform(key, shape=(sample_num, 4), minval=lb, maxval=ub)
    key2 = jax.random.split(key)[0]
    sample2 = jax.random.uniform(key2, shape=(sample_num, 4), minval=lb, maxval=ub)
    all_samples = jnp.concatenate([sample1, sample2], axis=0)

    min_loss_val = 1e12
    best_sample_p = all_samples[0]
    for p_candidate in all_samples:
        loss_arr = loss_fn(p_candidate, y_obs, p_candidate)
        loss_scalar = float(jnp.asarray(loss_arr).reshape(-1)[0])
        if loss_scalar < min_loss_val:
            min_loss_val = loss_scalar
            best_sample_p = p_candidate
    return best_sample_p

def hybrid_inversion(y_obs, use_auto=True):
    p_global_init = bayes_search(y_obs)
    p_opt, final_loss = lbfgs_optim(p_global_init, y_obs, use_auto_grad=use_auto)
    return p_opt, final_loss