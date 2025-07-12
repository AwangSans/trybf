import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import os
from scipy import signal
from obspy import read
import glob

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两个经纬度点之间的距离（单位：公里）
    使用Haversine公式
    """
    # 将经纬度转换为弧度
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球半径（公里）
    return c * r

def cross_correlation_timelag(data1, data2, sampling_rate):
    """
    计算两个时间序列的互相关和时间延迟
    注意：这是互相关时间延迟，不是真正的到时差
    真正的到时差需要识别同一个地震事件在不同台站的到达时间
    """
    # 确保数据长度相同
    min_length = min(len(data1), len(data2))
    data1 = data1[:min_length]
    data2 = data2[:min_length]
    
    # 计算互相关
    correlation = signal.correlate(data1, data2, mode='full')
    lags = signal.correlation_lags(len(data1), len(data2), mode='full')
    
    # 找到最大互相关值的位置
    max_idx = np.argmax(np.abs(correlation))
    max_cc_value = correlation[max_idx]
    timelag = lags[max_idx] / sampling_rate
    
    return max_cc_value, timelag

def main():
    # 读取台站信息
    station_info = {}
    with open('chuanxi_info.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                parts = line.strip().split()
                if len(parts) >= 3:
                    station_id = parts[0]
                    lat = float(parts[1])
                    lon = float(parts[2])
                    station_info[station_id] = {'lat': lat, 'lon': lon}
    
    print(f"Read {len(station_info)} station information")
    
    # 获取SAC文件列表
    sac_files = glob.glob('hour150_024hz/*.sac')
    print(f"Found {len(sac_files)} SAC files")
    
    # 存储所有台站的数据
    station_data = {}
    
    # 读取所有SAC文件
    for sac_file in sac_files:
        try:
            # 从文件名提取台站号
            filename = os.path.basename(sac_file)
            station_id = filename.split('.')[2]
            
            if station_id in station_info:
                # 读取SAC文件
                st = read(sac_file)
                tr = st[0]
                
                # 检查SAC头信息，看是否有到时信息
                print(f"Station {station_id} SAC header info:")
                print(f"  Sampling rate: {tr.stats.sampling_rate} Hz")
                print(f"  Start time: {tr.stats.starttime}")
                print(f"  End time: {tr.stats.endtime}")
                print(f"  SAC header keys: {list(tr.stats.sac.keys()) if hasattr(tr.stats, 'sac') else 'No SAC header'}")
                
                # 存储数据
                station_data[station_id] = {
                    'data': tr.data,
                    'sampling_rate': tr.stats.sampling_rate,
                    'file': sac_file,
                    'starttime': tr.stats.starttime,
                    'endtime': tr.stats.endtime
                }
                print(f"Loaded {station_id}: {len(tr.data)} samples, {tr.stats.sampling_rate} Hz")
        except Exception as e:
            print(f"Error reading {sac_file}: {e}")
    
    print(f"Successfully loaded {len(station_data)} station data")
    
    # 计算所有台站对之间的互相关
    distances = []
    timelags = []
    cc_values = []
    station_pairs = []
    
    station_ids = list(station_data.keys())
    
    for i, station1 in enumerate(station_ids):
        for j, station2 in enumerate(station_ids[i+1:], i+1):  # 避免重复计算
            try:
                # 获取台站数据
                data1 = station_data[station1]['data']
                data2 = station_data[station2]['data']
                sampling_rate = station_data[station1]['sampling_rate']
                
                # 计算互相关和时间延迟
                cc_value, timelag = cross_correlation_timelag(data1, data2, sampling_rate)
                
                # 只处理时间延迟为正的数据
                if timelag > 0:
                    # 计算台站间距离
                    lat1 = station_info[station1]['lat']
                    lon1 = station_info[station1]['lon']
                    lat2 = station_info[station2]['lat']
                    lon2 = station_info[station2]['lon']
                    
                    distance = haversine_distance(lat1, lon1, lat2, lon2)
                    
                    distances.append(distance)
                    timelags.append(timelag)
                    cc_values.append(cc_value)
                    station_pairs.append(f"{station1}-{station2}")
                    
            except Exception as e:
                print(f"Error processing {station1}-{station2}: {e}")
    
    print(f"Processed {len(distances)} station pair cross-correlations (timelag > 0)")
    
    # 转换为numpy数组
    distances = np.array(distances)
    timelags = np.array(timelags)
    cc_values = np.array(cc_values)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 创建散点图，颜色表示互相关值的大小
    scatter = plt.scatter(distances, timelags/100, c=np.abs(cc_values), cmap='viridis', 
                         alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cross-correlation Value (abs)', fontsize=12)
    
    # 添加速度线 (3, 4, 5, 6, 7, 8 km/s)
    if len(distances) > 0:
        max_distance = max(distances)
        distance_range = np.linspace(0, max_distance, 100)
        
        # 3 km/s 线
        timelag_3km = distance_range / 3.0
        plt.plot(distance_range, timelag_3km, 'r--', linewidth=2, label='3 km/s')
        
        # 4 km/s 线
        timelag_4km = distance_range / 4.0
        plt.plot(distance_range, timelag_4km, 'g--', linewidth=2, label='4 km/s')
        
        # 5 km/s 线
        timelag_5km = distance_range / 5.0
        plt.plot(distance_range, timelag_5km, 'b--', linewidth=2, label='5 km/s')
        
        # 6 km/s 线
        timelag_6km = distance_range / 6.0
        plt.plot(distance_range, timelag_6km, 'm--', linewidth=2, label='6 km/s')
        
        # 7 km/s 线
        timelag_7km = distance_range / 7.0
        plt.plot(distance_range, timelag_7km, 'c--', linewidth=2, label='7 km/s')
        
        # 8 km/s 线
        timelag_8km = distance_range / 8.0
        plt.plot(distance_range, timelag_8km, 'y--', linewidth=2, label='8 km/s')
    
    # 设置坐标轴标签和标题
    plt.xlabel('Station Distance (km)', fontsize=14)
    plt.ylabel('Time Lag (s)', fontsize=14)
    plt.title('Station Distance vs Time Lag (Calculated from SAC files)', fontsize=16, fontweight='bold')
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    if len(distances) > 0:
        plt.xlim(0, max(distances) * 1.05)
        plt.ylim(0, max(timelags/100) * 1.05)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('distance_timelag_from_sac.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印一些统计信息
    if len(distances) > 0:
        print(f"\nStatistics:")
        print(f"Distance range: {min(distances):.2f} - {max(distances):.2f} km")
        print(f"Time lag range: {min(timelags):.2f} - {max(timelags):.2f} s")
        print(f"Cross-correlation value range: {min(cc_values):.6f} - {max(cc_values):.6f}")
        
        # 计算相关系数
        correlation = np.corrcoef(distances, timelags)[0, 1]
        print(f"Correlation coefficient between distance and time lag: {correlation:.4f}")
        
        # 保存结果到文件
        with open('cross_correlation_results_from_sac.txt', 'w') as f:
            f.write("Station1\tStation2\tDistance(km)\tTimeLag(s)\tCC_Value\n")
            for i in range(len(distances)):
                f.write(f"{station_pairs[i]}\t{distances[i]:.2f}\t{timelags[i]:.6f}\t{cc_values[i]:.6f}\n")
        
        print(f"Results saved to cross_correlation_results_from_sac.txt")

if __name__ == "__main__":
    main() 