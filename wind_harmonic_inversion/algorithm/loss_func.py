import jax
import jax.numpy as jnp
from model.full_forward import forward_model

@jax.jit
def loss_fn(p, y_obs, p0, gamma=1e-5, lam_f=10.0):
    # 前向模型输出预测谐波 [fsh, Ish]
    y_pred = forward_model(p)
    f_pred, I_pred = y_pred[0], y_pred[1]
    f_obs, I_obs = y_obs[0], y_obs[1]
    # 相对误差，分母极小值防除0报错
    rel_err_I = (I_pred - I_obs) / (jnp.abs(I_obs) + 1e-8)
    rel_err_f = (f_pred - f_obs) / (jnp.abs(f_obs) + 1e-8)
    # 频率项权重放大10倍，强化偏航/桨距对应的频率约束
    loss_data = jnp.square(rel_err_I) + lam_f * jnp.square(rel_err_f)
    # 正则进一步降低，减少先验拉扯
    loss_reg = gamma * jnp.sum(jnp.square(p - p0))
    total_loss = loss_data + loss_reg
    return jnp.squeeze(total_loss)