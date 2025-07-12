import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import correlate, correlation_lags
import os

# 台站与文件名的映射
station_files = {
    '1827': '240825.000000.001827.EHZ_processed.sac',
    '1812': '240825.000000.001812.EHZ_processed.sac',
    '1038': '240825.000000.001038.EHZ_processed.sac',
}

folder = 'hour150_024hz'

# 读取数据
traces = {}
sampling_rate = None
min_length = None
for sta, fname in station_files.items():
    st = read(os.path.join(folder, fname))
    tr = st[0]
    traces[sta] = tr.data
    if sampling_rate is None:
        sampling_rate = tr.stats.sampling_rate
    else:
        assert abs(sampling_rate - tr.stats.sampling_rate) < 1e-6, 'Sampling rate mismatch!'
    if min_length is None:
        min_length = len(tr.data)
    else:
        min_length = min(min_length, len(tr.data))

# 截取最短长度，保证对齐
for sta in traces:
    traces[sta] = traces[sta][:min_length]

# 计算互相关
pairs = [('1827', '1812'), ('1827', '1038'), ('1812', '1038')]
cc_curves = {}
timelags = None
for sta1, sta2 in pairs:
    cc = correlate(traces[sta1], traces[sta2], mode='full', method='auto')
    lags = correlation_lags(len(traces[sta1]), len(traces[sta2]), mode='full')
    lags_sec = lags / sampling_rate
    cc_curves[(sta1, sta2)] = cc / (np.std(traces[sta1]) * np.std(traces[sta2]) * min_length)  # 归一化
    timelags = lags_sec  # 三组长度一致

# 绘制三个子图
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
pair_titles = [
    'Cross-correlation: 1827 & 1812',
    'Cross-correlation: 1827 & 1038',
    'Cross-correlation: 1812 & 1038',
]
colors = ['r', 'g', 'b']
for i, ((sta1, sta2), cc) in enumerate(cc_curves.items()):
    axes[i].plot(timelags, cc, color=colors[i])
    axes[i].set_ylabel('CC')
    axes[i].set_title(pair_titles[i])
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel('Timelag (s)')
plt.tight_layout()
plt.savefig('cc_curve_three_stations_subplots.png', dpi=300)
plt.show() 