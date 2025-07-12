import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')


def calculate_snr(fk, nsamp, dt, cap_find):
    # 计算频率分辨率
    df = 1 / (nsamp * dt)
    # 找到最大值的索引
    max_index = np.unravel_index(np.argmax(fk), fk.shape)
    # 计算目标信号的中心频率
    center_freq = cap_find / (nsamp * dt)
    # 确定目标信号的频率范围
    signal_freq_min = center_freq - 0.02
    signal_freq_max = center_freq + 0.02
    # 计算对应的频率索引范围
    signal_index_min = int(signal_freq_min / df)
    signal_index_max = int(signal_freq_max / df)
    # 提取目标信号
    signal = fk[:, signal_index_min:signal_index_max]
    # 提取噪声（排除信号区域）
    noise = np.delete(fk, np.s_[signal_index_min:signal_index_max], axis=1)
    # 计算信号和噪声的功率
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    # 计算信噪比
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def load_and_plot(fk_file, params):
    # 从.npy文件中加载fk
    fk = np.load(fk_file)

    fk = 10 * np.log10(fk / fk.max())
    # 计算信噪比
    snr = calculate_snr(fk, params['nsamp'], params['dt'], params['cap_find'])
    # 提取参数
    smin = params['smin']
    smax = params['smax']
    cap_find = params['cap_find']
    cap_fave = params['cap_fave']
    nsamp = params['nsamp']
    dt = params['dt']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(fk.T, extent=[smin, smax, smax, smin],
                   cmap='gist_stern_r', interpolation='none')
    plt.title('Slowness Spectrum at %.03f +- %.03f[Hz], SNR: %.2f dB' % (
        cap_find / (nsamp * dt), cap_fave / (nsamp * dt), snr))
    ax.set_xlim([smin, smax])
    ax.set_ylim([smin, smax])
    ax.set_xlabel('East/West Slowness [s/deg]')
    ax.set_ylabel('North/South Slowness [s/deg]')

    circle = plt.Circle((0, 0), np.lib.scimath.sqrt(
        (0.3 * 111.19) ** 2), color='w', fill=False, alpha=0.4)
    plt.gcf().gca().add_artist(circle)

    circle = plt.Circle((0, 0), np.lib.scimath.sqrt(
        (0.24 * 111.19) ** 2), color='w', fill=False, alpha=0.4)
    plt.gcf().gca().add_artist(circle)

    cbar = fig.colorbar(im)
    cbar.set_label('absolute power', rotation=270)

    # 保存图像而不是显示它
    img_name = os.path.basename(fk_file).split('.')[0] + '.png'
    img_path = os.path.join('./fk_fig_test', img_name)
    print(img_path)
    plt.savefig(img_path)
    plt.close()


def process_directory(directory, params):
    if not os.path.exists(directory) or not os.path.isdir(directory):
        print(f"Error: {directory} does not exist or is not a directory.")
        return

    save_dir = './fk_fig_test'
    os.makedirs(save_dir, exist_ok=True)

    files = [os.path.join(directory, f)
             for f in os.listdir(directory) if f.endswith('.npy')]

    for fk_file in tqdm(files):
        load_and_plot(fk_file, params)


# 设置你的参数
params = {
    'smin': -20,
    'smax': 20,
    'cap_find': 120,
    'cap_fave': 20,
    'nsamp': 600,
    'dt': 1
}

# 调用函数处理指定目录
process_directory('./fk_body_wave/', params)
    