import os
import obspy
import matplotlib.pyplot as plt


# 定义输出目录
output_dir = '.\\output'

# 获取目录下所有.sac文件
sac_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.sac')]

# 计算子图布局
num_files = len(sac_files)
nrows = 1
ncols = num_files
if num_files > 4:
    # 若文件数量大于 4，调整为多行布局
    nrows = (num_files + 3) // 4
    ncols = 4

# 创建一个包含多个子图的图形
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

# 确保 axes 是二维数组（即使只有一行）
if nrows == 1:
    axes = axes.reshape(1, -1)

# 遍历每个.sac文件
for i, sac_file in enumerate(sac_files):
    try:
        # 读取.sac文件
        st = obspy.read(sac_file)
        tr = st[0]  # 假设每个文件只有一个波形数据

        # 获取时间和振幅数据
        time = tr.times()
        amplitude = tr.data

        # 计算子图位置
        row = i // ncols
        col = i % ncols

        # 在对应的子图中绘制波形图
        axes[row, col].plot(time, amplitude)
        axes[row, col].set_title(os.path.basename(sac_file))
        axes[row, col].set_xlabel('时间 (s)')
        axes[row, col].set_ylabel('振幅')

    except Exception as e:
        print(f"读取文件 {sac_file} 时出现错误: {e}")

# 隐藏多余的子图
for i in range(num_files, nrows * ncols):
    row = i // ncols
    col = i % ncols
    axes[row, col].axis('off')

# 调整子图布局
plt.tight_layout()

# 显示图形
plt.show()
    