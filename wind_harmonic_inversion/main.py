import os
import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
from model.full_forward import forward_model, batch_forward
from algorithm.hybrid_opt import hybrid_inversion
from utils.param_config import get_param_bounds
from utils.plot_tool import plot_convergence, plot_error_bar

if not os.path.exists("output"):
    os.makedirs("output")

key_root = jax.random.PRNGKey(0)
bounds = get_param_bounds()
lb, ub = bounds[:,0], bounds[:,1]

sample_num = 100
key = jax.random.PRNGKey(123)
p_true_set = jax.random.uniform(key, (sample_num,4), minval=lb, maxval=ub)
y_clean = batch_forward(p_true_set)
noise = jax.random.normal(key, y_clean.shape) * 0.01
y_obs_set = y_clean + noise

err_auto_total = jnp.zeros(4)
err_fd_total = jnp.zeros(4)
time_auto_list = []
time_fd_list = []

for idx in range(sample_num):
    p_t = p_true_set[idx]
    y_o = y_obs_set[idx]
    # 删掉固定全局p0，不再定义、不再传入
    # AD反演，去掉p0传参
    t1 = time.time()
    p_ad, _ = hybrid_inversion(y_o, use_auto=True)
    t_auto = time.time() - t1
    time_auto_list.append(t_auto)
    err_auto_total += jnp.abs(p_ad - p_t) / jnp.abs(p_t) * 100
    # FD反演，去掉p0传参
    t2 = time.time()
    p_fd, _ = hybrid_inversion(y_o, use_auto=False)
    t_fd = time.time() - t2
    time_fd_list.append(t_fd)
    err_fd_total += jnp.abs(p_fd - p_t) / jnp.abs(p_t) * 100

err_auto_mean = err_auto_total / sample_num
err_fd_mean = err_fd_total / sample_num
param_names = ["轮毂风速", "风切变指数", "偏航角", "桨距角"]

plt.figure(figsize=(10,4))
plot_error_bar(err_auto_mean, param_names, "自动微分")
plt.savefig("output/error_auto.png")
plt.clf()

plt.figure(figsize=(10,4))
plot_error_bar(err_fd_mean, param_names, "有限差分")
plt.savefig("output/error_fd.png")
plt.clf()

print(f"自动微分单样本平均耗时：{jnp.mean(jnp.array(time_auto_list)):.4f} s")
print(f"有限差分单样本平均耗时：{jnp.mean(jnp.array(time_fd_list)):.4f} s")
print(f"AD平均反演误差：{err_auto_mean}")
print(f"FD平均反演误差：{err_fd_mean}")