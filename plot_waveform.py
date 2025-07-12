import numpy as np
import matplotlib.pyplot as plt
from obspy import read

# 读取SAC文件
sac_file = "data/240825.000000.001038.EHZ.sac"
st = read(sac_file)

# 获取第一个轨迹
tr = st[0]

# 获取时间和数据
time = np.arange(len(tr.data)) * tr.stats.delta
data = tr.data

# 创建图形
plt.figure(figsize=(12, 6))
plt.plot(time[0:10000], data[0:10000], 'b-', linewidth=0.5)

# 设置标签和标题（使用英文）
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title(f'Waveform - {tr.stats.station}.{tr.stats.channel}')

# 添加网格
plt.grid(True, alpha=0.3)

# 保存图片
plt.tight_layout()
plt.show()
plt.close()

# 打印基本信息
print(f"Station: {tr.stats.station}")
print(f"Channel: {tr.stats.channel}")
print(f"Sampling rate: {tr.stats.sampling_rate} Hz")
print(f"Number of data points: {len(tr.data)}")
print(f"Total duration: {tr.stats.endtime - tr.stats.starttime:.2f} seconds")
print("Waveform plot saved as 'waveform_plot.png'") 