import numpy as np
import nolds

ts_3d = np.load("results_lstm_24.8.npy")

m = 3
tau = 1

# 对每个维度分别计算李雅普诺夫指数
for i in range(ts_3d.shape[1]):
    ts = ts_3d[:, i]
    LEs = nolds.lyap_r(ts, emb_dim=m, lag=tau)
    print(f'第{i+1}维的李雅普诺夫指数为：', LEs)