import jax
import jax.numpy as jnp
from algorithm.loss_func import loss_fn
grad_auto = jax.grad(loss_fn)
# 扰动放大至1e-4，适配ODE动力学输出小幅变化特性
def grad_fd(p, y_obs, p0, eps=1e-4):
    p_jax = jnp.array(p)
    grad = jnp.zeros_like(p_jax)
    for i in range(len(p_jax)):
        # 自适应相对扰动，下限防0参数
        delta = jnp.maximum(jnp.abs(p_jax[i]) * eps, 1e-8)
        p_plus = p_jax.at[i].set(p_jax[i] + delta)
        loss_p = loss_fn(p_plus, y_obs, p0)
        p_minus = p_jax.at[i].set(p_jax[i] - delta)
        loss_m = loss_fn(p_minus, y_obs, p0)
        grad = grad.at[i].set((loss_p - loss_m) / (2 * delta))
    return jnp.asarray(grad)