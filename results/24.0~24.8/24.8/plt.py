import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 从npy文件中加载数据
data1 = np.load('val_loss_cls.npy')
data2 = np.load('val_loss.npy')

# 创建x轴数据
x = np.arange(0, 100)

# 创建一个新的图形
plt.figure(figsize=(5, 4))

# 使用seaborn绘制曲线
sns.lineplot(x=x, y=data1, color='blue', label='class')
sns.lineplot(x=x, y=data2, color='red', label='hybrid')

# 添加x轴和y轴的标签
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('rho=24.8')
# 显示图形
# 保存图形为图片文件
plt.savefig('graph.png', dpi=300)