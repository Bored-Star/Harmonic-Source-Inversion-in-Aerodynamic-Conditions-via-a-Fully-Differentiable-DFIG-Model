import jax.numpy as jnp
import matplotlib.pyplot as plt
# 解决中文方框
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 修改入参逻辑：直接传入一维误差数组，不再拆true/pred
def plot_error_bar(err_arr, param_names, title_str):
    plt.figure()
    plt.bar(param_names, err_arr)
    plt.ylabel("相对误差 (%)")
    plt.title(f"{title_str} 参数相对误差")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_convergence(loss_history, title_str):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("迭代步数")
    plt.ylabel("损失函数值")
    plt.title(f"{title_str} 收敛曲线")
    plt.grid(True, alpha=0.3)
    plt.show()