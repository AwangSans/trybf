import os
import glob
import numpy as np
import obspy
from obspy import read
from itertools import combinations

def process_sac_files(directory, output_file):
    # 获取目录下所有.sac文件
    sac_files = glob.glob(os.path.join(directory, "*.sac"))
    
    if not sac_files:
        print("未找到SAC文件")
        return
    
    # 读取所有SAC文件
    traces = []
    file_names = []
    for sac_file in sac_files:
        try:
            st = read(sac_file)
            traces.append(st[0])
            file_names.append(os.path.basename(sac_file))
        except Exception as e:
            print(f"读取文件 {sac_file} 时出错: {e}")
    
    # 计算两两之间的互相关
    results = []
    for i, j in combinations(range(len(traces)), 2):
        tr1 = traces[i]
        tr2 = traces[j]
        
        # 确保采样率相同
        if tr1.stats.sampling_rate != tr2.stats.sampling_rate:
            tr2.resample(tr1.stats.sampling_rate)
        
        # 数据对齐（假设起始时间相同）
        npts1 = len(tr1.data)
        npts2 = len(tr2.data)
        npts = min(npts1, npts2)
        
        # 计算互相关
        cc = np.correlate(tr1.data[:npts], tr2.data[:npts], mode='same')
        
        # 找到最大互相关值的位置
        max_idx = np.argmax(cc)
        if cc[max_idx] <= 0:
            continue
        
        # 计算时间延迟（以秒为单位）
        npts_cc = len(cc)
        midpoint = (npts_cc - 1) // 2
        timelag = (max_idx - midpoint) / tr1.stats.sampling_rate
        
        # 记录结果
        results.append({
            'file1': file_names[i],
            'file2': file_names[j],
            'max_cc_value': cc[max_idx],
            'timelag': timelag
        })
    
    # 输出结果到文本文件
    with open(output_file, 'w') as f:
        f.write("File1\tFile2\tMax_CC_Value\tTimeLag(s)\n")
        for result in results:
            f.write(f"{result['file1']}\t{result['file2']}\t{result['max_cc_value']:.6f}\t{result['timelag']:.6f}\n")
    
    print(f"结果已保存到 {output_file}")

if __name__ == "__main__":
    # 请修改为实际的目录路径
    directory = "hour150_024hz"
    # 输出结果的文件名
    output_file = "cross_correlation_results.txt"
    
    process_sac_files(directory, output_file)    