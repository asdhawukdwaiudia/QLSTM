import numpy as np

def logistic_map(x, mu):
    return mu * x * (1 - x)

def lyapunov_exponent(mu, x0, n):
    x = x0
    sum = 0
    epsilon = 1e-10  # 添加一个小的正数以防止除以 0
    for i in range(n):
        sum += np.log(abs(mu - 2*mu*x + epsilon))
        x = logistic_map(x, mu)
    return sum / n

def calculate(mu):
    # 参数设置
    x0 = 0.5  # 初始值
    n = 10000  # 迭代次数

    # 计算李雅普诺夫指数
    lyap_exp = lyapunov_exponent(mu, x0, n)
    print(mu,lyap_exp)

a = np.linspace(3,4,100)

for i in range(100):
    calculate(a[i])