import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import pandas as pd

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
    
    # 读取互相关结果
    distances = []
    timelags = []
    cc_values = []
    
    with open('cross_correlation_results.txt', 'r', encoding='utf-8') as f:
        next(f)  # 跳过标题行
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    file1 = parts[0]
                    file2 = parts[1]
                    cc_value = float(parts[2])
                    timelag = float(parts[3])
                    
                    # 只处理时间延迟为正的数据
                    if timelag > 0:
                        # 从文件名提取台站号
                        station1 = file1.split('.')[2]  # 提取台站号部分
                        station2 = file2.split('.')[2]
                        
                        # 检查台站是否在信息文件中
                        if station1 in station_info and station2 in station_info:
                            # 计算台站间距离
                            lat1 = station_info[station1]['lat']
                            lon1 = station_info[station1]['lon']
                            lat2 = station_info[station2]['lat']
                            lon2 = station_info[station2]['lon']
                            
                            distance = haversine_distance(lat1, lon1, lat2, lon2)
                            
                            distances.append(distance)
                            timelags.append(timelag)
                            cc_values.append(cc_value)
    
    print(f"Processed {len(distances)} station pair cross-correlation results (timelag > 0)")
    
    # 转换为numpy数组
    distances = np.array(distances)
    timelags = np.array(timelags)
    cc_values = np.array(cc_values)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 创建散点图，颜色表示互相关值的大小
    scatter = plt.scatter(distances, timelags, c=cc_values, cmap='viridis', 
                         alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cross-correlation Value', fontsize=12)
    
    # 添加速度线 (3, 4, 5 km/s)
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
    
    # 设置坐标轴标签和标题
    plt.xlabel('Station Distance (km)', fontsize=14)
    plt.ylabel('Time Lag (s)', fontsize=14)
    plt.title('Station Distance vs Time Lag (Positive Values Only)', fontsize=16, fontweight='bold')
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    plt.xlim(0, max(distances) * 1.05)
    plt.ylim(0, max(timelags) * 1.05)
    
    # 添加统计信息
    plt.text(0.02, 0.98, f'Data points: {len(distances)}', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('distance_timelag_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印一些统计信息
    print(f"\nStatistics:")
    print(f"Distance range: {min(distances):.2f} - {max(distances):.2f} km")
    print(f"Time lag range: {min(timelags):.2f} - {max(timelags):.2f} s")
    print(f"Cross-correlation value range: {min(cc_values):.6f} - {max(cc_values):.6f}")
    
    # 计算相关系数
    correlation = np.corrcoef(distances, timelags)[0, 1]
    print(f"Correlation coefficient between distance and time lag: {correlation:.4f}")

if __name__ == "__main__":
    main() 