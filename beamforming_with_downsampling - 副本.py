from obspy import read, Stream
import numpy as np
import scipy as sp
from numba import jit
import multiprocessing
import os
import glob
import math as m

def calculate_snr(fk, nsamp, dt, cap_find):
    """
    计算信噪比，基于FK谱中最大值周围0.02Hz的频率范围
    """
    # 计算频率分辨率
    df = 1.0 / (nsamp * dt)
    # 中心频率
    center_freq = cap_find / (nsamp * dt)
    # 信号频率范围
    signal_freq_min = center_freq - 0.02
    signal_freq_max = center_freq + 0.02
    # 转换为频率索引（假设FK谱的频率轴是隐含的，这里简化为一维处理）
    # 实际应用中需根据FK谱的频率维度调整
    freq_axis = np.linspace(0, 1/(2*dt), fk.shape[0])  # 示例频率轴，需根据实际情况修改
    signal_mask = (freq_axis >= signal_freq_min) & (freq_axis <= signal_freq_max)
    
    # 提取信号和噪声功率
    signal_power = np.mean(fk[signal_mask] ** 2)
    noise_power = np.mean(fk[~signal_mask] ** 2)
    
    # 计算SNR
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def grt(r1, r2, s1, s2):
    slat = s1 * np.pi / 180.
    slon = s2 * np.pi / 180.
    elat = r1 * np.pi / 180.
    elon = r2 * np.pi / 180.

    slat = m.atan(.996647 * m.tan(slat))
    elat = m.atan(.996647 * m.tan(elat))

    slat = np.pi / 2.0 - slat
    elat = np.pi / 2.0 - elat

    if slon < 0.0:
        slon += 2.0 * np.pi
    if elon < 0.0:
        elon += 2.0 * np.pi
    a = m.sin(elat) * m.cos(elon)
    b = m.sin(elat) * m.sin(elon)
    c = m.cos(elat)
    a1 = m.sin(slat) * m.cos(slon)
    b1 = m.sin(slat) * m.sin(slon)
    c1 = m.cos(slat)

    cd = a * a1 + b * b1 + c * c1

    if cd > 1.0:
        cd = 1.0
    if cd < -1.0:
        cd = -1.0
    decl = m.acos(cd) * 180.0 / m.pi
    dist = decl * np.pi * 6371.0 / 180.0

    tmp1 = m.cos(elon) * m.cos(slon) + m.sin(elon) * m.sin(slon)
    tmp2a = 1.0 - cd * cd

    if tmp2a <= 0.0:
        tmp2 = 0.0
        tmp3 = 1.0
    else:
        tmp2 = m.sqrt(tmp2a)
        tmp3 = (m.sin(elat) * m.cos(slat) - m.cos(elat) * m.sin(slat) * tmp1) / tmp2

    if tmp3 > 1.0:
        tmp3 = 1.0
    if tmp3 < -1.0:
        tmp3 = -1.0
    z = m.acos(tmp3)

    if (m.sin(slon) * m.cos(elon) - m.cos(slon) * m.sin(elon)) < 0.0:
        z = 2.0 * m.pi - z

    az = 180.0 * z / m.pi

    tmp1 = m.cos(slon) * m.cos(elon) + m.sin(slon) * m.sin(elon)
    tmp2a = 1.0 - cd * cd
    if tmp2a <= 0.0:
        tmp2 = 0.0
        tmp3 = 1.0
    else:
        tmp2 = m.sqrt(tmp2a)
        tmp3 = (m.sin(slat) * m.cos(elat) - m.cos(slat) * m.sin(elat) * tmp1) / tmp2

    if tmp3 > 1.0:
        tmp3 = 1.0
    if tmp3 < -1.0:
        tmp3 = -1.0

    bz = m.acos(tmp3)

    if (m.sin(elon) * m.cos(slon) - m.cos(elon) * m.sin(slon)) < 0.0:
        bz = 2.0 * m.pi - bz

    baz = 180.0 * bz / m.pi

    return decl, dist, az, baz


def get_metadata2(meta_f):
    d = {}
    with open(meta_f, 'r') as f:
        for line in f:
            x = line.split()
            if len(x) >= 3:
                d[x[0]] = x[1], x[2]
    return d


def metric_mseed(st, d, nr):
    rx_0, ry_0 = d[st[0].stats.station]
    rx = np.zeros(nr)
    ry = np.zeros(nr)
    for i in range(nr):
        rx_i, ry_i = d[st[i].stats.station]
        decl, dist, az, baz = grt(float(rx_0), float(ry_0), float(rx_i), float(ry_i))
        rx[i] = decl * sp.cos(0.017453 * (90.0 - az))
        ry[i] = decl * sp.sin(0.017453 * (90.0 - az))
    return rx, ry


@jit(nopython=True)
def compute_s_matrices(nwin, nr, find, fave, rft, ift):
    smr = np.zeros((2 * fave + 1, nr, nr))
    smi = np.zeros((2 * fave + 1, nr, nr))

    for n in range(nwin):
        for i in range(nr):
            for j in range(nr):
                for l in range(find - fave, find + fave + 1):
                    smr[l - find + fave, i, j] += rft[n, i, l] * rft[n, j, l] + ift[n, i, l] * ift[n, j, l]
                    smi[l - find + fave, i, j] += rft[n, j, l] * ift[n, i, l] - rft[n, i, l] * ift[n, j, l]

    return smr, smi


@jit(nopython=True)
def compute_fk(ismr, ismi, nk, rx, ry, kinc, kmin, nr, nsamp, delta, smin, sinc, fave, find):
    fk = np.zeros((nk, nk))
    tfk = np.zeros((nk, nk))

    for g in range(find - fave, find + fave + 1):
        freq = g / (nsamp * delta)
        kmin = 2 * np.pi * freq * smin
        kinc = 2 * np.pi * freq * sinc
        for i in range(nk):
            kx = -(kmin + i * kinc)
            for j in range(nk):
                ky = -(kmin + j * kinc)
                fk[i, j] = 0.0
                for m in range(nr):
                    fk[i, j] += ismr[g - find + fave, m, m]
                    for n in range(m + 1, nr):
                        arg = kx * (rx[m] - rx[n]) + ky * (ry[m] - ry[n])
                        fk[i, j] += 2.0 * (ismr[g - find + fave, m, n] * np.cos(arg) - ismi[g - find + fave, m, n] * np.sin(arg))
                fk[i, j] = 1.0 / fk[i, j]
        tfk += fk

    return tfk


def IAS_Capon(nsamp, nr, rx, ry, st, smin, smax, sinc, find, fave, delta, dl, overlap, taper):

    freq = find / (nsamp * delta)
    df = fave / (nsamp * delta)
    print('IAS Capon DOA estimation is performed at:', 'freq', freq, '+-', df)
    kmin = 2 * sp.pi * smin * freq
    kmax = 2 * sp.pi * smax * freq
    kinc = 2 * sp.pi * sinc * freq
    nk = int(((kmax - kmin) / kinc + 0.5) + 1)

    if overlap:
        nwin = int(np.array(st[0].stats.npts / nsamp)) * 2 - 1
        xt = np.zeros((nr, nwin, nsamp))
        for i in range(nr):
            for j in range(nwin):
                xt[i][j] = st[i][j * nsamp // 2:(j + 2) * nsamp // 2]
                xt[i][j] -= np.mean(xt[i][j])
                if taper:
                    xt[i][j] *= np.hanning(nsamp)
    else:
        nwin = int(np.array(st[0].stats.npts / nsamp))
        xt = np.zeros((nr, nwin, nsamp))
        for i in range(nr):
            for j in range(nwin):
                xt[i][j] = st[i][j * nsamp:(j + 1) * nsamp]
                xt[i][j] -= np.mean(xt[i][j])
                if taper:
                    xt[i][j] *= np.hanning(nsamp)

    smr = np.zeros((2 * fave + 1, nr, nr))
    smi = np.zeros((2 * fave + 1, nr, nr))

    rft = np.zeros((nwin, nr, nsamp // 2 + 1))
    ift = np.zeros((nwin, nr, nsamp // 2 + 1))
    for i in range(nwin):
        for j in range(nr):
            tp = np.fft.rfft(xt[j][i], nsamp)
            rft[i][j] = tp.real
            ift[i][j] = tp.imag

    smr, smi = compute_s_matrices(nwin, nr, find, fave, rft, ift)
    smr /= nwin
    smi /= nwin

    fw = 0.0
    fe = 0.0
    for m in range(2 * fave + 1):
        wmean = 0.0
        w = np.zeros(nr)
        for i in range(nr):
            w[i] = (smr[m][i][i] * smr[m][i][i] + smi[m][i][i] * smi[m][i][i]) ** (-0.25)
            wmean += 1.0 / (w[i] ** 2)
            fw += 1.0 / (w[i] ** 2)
            fe += np.abs(smr[m][i][i] + 1j * smi[m][i][i]) ** 2
        for i in range(nr):
            for j in range(nr):
                smr[m][i][j] *= w[i] * w[j]
                smi[m][i][j] *= w[i] * w[j]
    fw /= nr * (2 * fave + 1)
    fe /= nr

    print('Diagonal Loading On!')
    mi = np.identity(nr)
    for i in range(2 * fave + 1):
        smr[i] += mi * smr[i].trace() / (nsamp) * dl

    tx = smr + 1j * smi
    itx = np.zeros((2 * fave + 1, nr, nr), dtype=complex)
    for m in range(2 * fave + 1):
        itx[m] = np.linalg.inv(tx[m])
    ismr = itx.real
    ismi = itx.imag

    fk = np.zeros((nk, nk))

    fk = compute_fk(ismr, ismi, nk, rx, ry, kinc, kmin, nr,
                    nsamp, delta, smin, sinc, fave, find)

    return fk.real


def single_saclist_process(arg_list):
    sac_list_file = arg_list[0]
    sta_coord_file = arg_list[1]
    fk_file = arg_list[2]

    nsamp = arg_list[3]
    smin = arg_list[4]
    smax = arg_list[5]
    sinc = arg_list[6]
    cap_find = arg_list[7]
    cap_fave = arg_list[8]
    dl = arg_list[9]

    st = Stream()
    with open(sac_list_file, 'r') as f:
        for line in f:
            file_path = line.strip()
            st += read(file_path)

    dic_meta = get_metadata2(sta_coord_file)

    nr = st.count()
    dt = st[0].stats.delta
    rx, ry = metric_mseed(st, dic_meta, nr)

    fk = IAS_Capon(nsamp, nr, rx, ry, st, smin, smax, sinc, cap_find, cap_fave, dt, dl, overlap=True, taper=True)
    # 计算信噪比
    snr = calculate_snr(fk, nsamp, dt, cap_find)
    print(f"File: {fk_file}, SNR: {snr:.2f} dB")  # 打印SNR结果

    np.save(fk_file, fk)


import os
import obspy
import numpy as np
from scipy import signal


def manual_decimate(trace, target_rate):
    original_rate = trace.stats.sampling_rate
    factor = int(original_rate / target_rate)
    if factor <= 16:
        trace.decimate(factor=factor)
    else:
        while factor > 1:
            if factor > 16:
                current_factor = 16
            else:
                current_factor = factor
            trace.decimate(factor=current_factor)
            factor = factor // current_factor
    return trace


def read_and_downsample_sac(file_path, target_rate):
    try:
        st = obspy.read(file_path)
        for tr in st:
            tr = manual_decimate(tr, target_rate)
        return st
    except Exception as e:
        print(f"读取或处理 SAC 数据时出错: {e}")
        return None


def process_with_time_window(downsampled_stream):
    window_length = 7 * 60
    processed_stream = obspy.Stream()
    for tr in downsampled_stream:
        data = tr.data
        num_windows = len(data) // window_length
        new_data = []
        for i in range(num_windows):
            start_idx = i * window_length
            end_idx = start_idx + window_length
            window_data = data[start_idx:end_idx]
            mean_val = np.mean(window_data)
            std_val = np.std(window_data)
            half_std = 0.5 * std_val
            upper_bound = mean_val + half_std
            lower_bound = mean_val - half_std
            window_data = np.where(window_data > upper_bound, upper_bound, window_data)
            window_data = np.where(window_data < lower_bound, lower_bound, window_data)
            new_data.extend(window_data)
        if len(data) % window_length != 0:
            start_idx = num_windows * window_length
            window_data = data[start_idx:]
            mean_val = np.mean(window_data)
            std_val = np.std(window_data)
            half_std = 0.5 * std_val
            upper_bound = mean_val + half_std
            lower_bound = mean_val - half_std
            window_data = np.where(window_data > upper_bound, upper_bound, window_data)
            window_data = np.where(window_data < lower_bound, lower_bound, window_data)
            new_data.extend(window_data)
        new_trace = tr.copy()
        new_trace.data = np.array(new_data)
        new_trace.stats.npts = len(new_data)
        processed_stream.append(new_trace)
    return processed_stream


def pad_traces_to_max_length(streams):
    max_npts = 0
    for stream in streams:
        for trace in stream:
            if trace.stats.npts > max_npts:
                max_npts = trace.stats.npts
    for stream in streams:
        for trace in stream:
            npts = trace.stats.npts
            if npts < max_npts:
                padding = np.zeros(max_npts - npts)
                trace.data = np.concatenate((trace.data, padding))
                trace.stats.npts = max_npts
    return streams


def process_all_files_in_folder(folder_path, start_hour, target_rate, output_folder):
    all_processed_streams = []
    three_hours_points = 3 * 60 * 60 * target_rate
    start_points = start_hour * 60 * 60 * target_rate
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.sac'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                downsampled_stream = read_and_downsample_sac(file_path, target_rate)
                if downsampled_stream:
                    for tr in downsampled_stream:
                        if len(tr.data) > start_points + three_hours_points:
                            tr.data = tr.data[start_points:start_points + three_hours_points]
                            tr.stats.npts = three_hours_points

                    processed_stream = process_with_time_window(downsampled_stream)
                    all_processed_streams.append(processed_stream)

                    file_name = os.path.splitext(os.path.basename(file_path))[0] + '_processed.sac'
                    processed_file_path = os.path.join(output_folder, file_name)
                    processed_stream.write(processed_file_path, format="SAC")

    all_processed_streams = pad_traces_to_max_length(all_processed_streams)


if __name__ == '__main__':
    nsamp = 600
    smin = -50.0
    smax = 50.0
    sinc = 0.25
    cap_find = int(0.24*nsamp)
    cap_fave = 6
    dl = 1

    for hour in range(12*0,12*22):
        folder_path = "data"
        start_hour = hour * 2
        target_rate = 1
        output_folder = 'output'
        process_all_files_in_folder(folder_path, start_hour, target_rate, output_folder)

        sac_list_dir = 'C:/Users/Awang/Desktop/学科交叉/trybf/saclist_by_time'
        coord_list_file = './chuanxi_info.txt'
        output_dir = './fk_body_wave'

        os.makedirs(output_dir, exist_ok=True)

        arg_list_list = []

        for sac_list in glob.glob(sac_list_dir + '/*.txt'):
            sac_list_file_name = os.path.basename(sac_list)
            label = sac_list_file_name.split('.')[0]
            # 将当前小时信息加入输出文件名
            output_file = os.path.join(output_dir, f'{label}_hour_{hour}_fk.npy')

            arg_list = [sac_list, coord_list_file, output_file,
                        nsamp, smin, smax, sinc, cap_find, cap_fave, dl]

            arg_list_list.append(arg_list)

        for arg_list in arg_list_list:
            single_saclist_process(arg_list)
    