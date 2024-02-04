import numpy as np
import os
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee'])
folders = [str(round(i,2)) for i in np.arange(24.0, 25.0, 0.2)]
files = [ 'val_loss_lstm.npy','val_loss.npy','val_loss_cls.npy']
#plt.figure(figsize=(5, 1),dpi=100)
data_dict = {}

for folder in folders:
    for file in files:
        path = os.path.join(folder, file)
        if os.path.exists(path):
            if file == 'val_loss_lstm.npy':
                data = np.load(path)
                data_dict[path] = {'data':data, 'label':folder,'type':"lstm"}
            elif file == 'val_loss.npy':
                data = np.load(path)
                data_dict[path] = {'data':data, 'label':folder,'type':"quan"}
            elif file == 'val_loss_cls.npy':
                data = np.load(path)
                data_dict[path] = {'data':data, 'label':folder,'type':"ablation"} 

fig, axs = plt.subplots(1, 5, sharey=True, figsize=(7.5, 1.45), dpi=150)
# fig.text(0.04, 0.5, 'Validation Set Loss', va='center', rotation='vertical')
fig.text(0.06, 0.5, 'Validation set loss', va='center', rotation='vertical', fontsize=10)
# for i, (path, data) in enumerate(data_dict.items()):
#     # 绘制数据点
#     axs[i // 2].plot(data['data'], label='{}'.format('ablation' if data['type'] == 'ablation' else 'quantum'),
#              linewidth=0.5)
#     axs[i // 2].set_title('$\\rho = {}$'.format(data['label']))
#     axs[i // 2].legend()
#     axs[i // 2].set_xlabel("Epoch")
for i, (path, data) in enumerate(data_dict.items()):
    label = 'Ablation' if data['type'] == 'ablation' else ('QLSTM' if data['type'] == 'quan' else 'LSTM')  # 添加 'lstm'
    axs[i // 3].plot(data['data'], label=label, linewidth=0.5)
    axs[i // 3].set_title('$\\rho = {}$'.format(data['label']))
    axs[i // 3].legend()
    axs[i // 3].set_xlabel("Epoch") 

# 添加图例

# 显示图形
# plt.show()
plt.savefig('validation_set_loss_v2.pdf', bbox_inches='tight')