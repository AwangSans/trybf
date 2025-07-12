import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_station_info(filename):
    """加载台站信息"""
    stations = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                station_id = parts[0]
                lat = float(parts[1])
                lon = float(parts[2])
                elevation = float(parts[3])
                stations[station_id] = {
                    'lat': lat,
                    'lon': lon,
                    'elevation': elevation
                }
    return stations

def load_cc_results(filename):
    """加载互相关结果"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过标题行
            parts = line.strip().split('\t')
            if len(parts) >= 4:  # 修正：只需要4个字段
                station_pair = parts[0]
                distance = float(parts[1])
                timelag = float(parts[2])
                cc_value = float(parts[3])
                
                # 解析台站对
                stations = station_pair.split('-')
                if len(stations) == 2:
                    data.append({
                        'station1': stations[0],
                        'station2': stations[1],
                        'distance': distance,
                        'timelag': timelag,
                        'cc_value': cc_value
                    })
    return pd.DataFrame(data)

def plot_three_stations_cc():
    """绘制三个台站的互相关图"""
    
    # 加载数据
    stations = load_station_info('chuanxi_info.txt')
    cc_data = load_cc_results('cross_correlation_results_from_sac.txt')
    
    # 目标台站
    target_stations = ['001827', '001812', '001038']
    station_names = {'001827': '1827', '001812': '1812', '001038': '1038'}
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('台站1827、1812、1038互相关分析', fontsize=16, fontweight='bold')
    
    # 1. 台站位置图
    ax1.set_title('台站位置分布', fontsize=14, fontweight='bold')
    
    # 绘制所有台站
    for station_id, info in stations.items():
        if station_id in target_stations:
            # 目标台站用大圆点标记
            ax1.scatter(info['lon'], info['lat'], s=200, c='red', 
                       marker='o', edgecolors='black', linewidth=2, zorder=5)
            ax1.annotate(station_names[station_id], 
                        (info['lon'], info['lat']), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12, fontweight='bold')
        else:
            # 其他台站用小圆点标记
            ax1.scatter(info['lon'], info['lat'], s=50, c='lightblue', 
                       alpha=0.6, marker='o')
    
    # 连接目标台站
    target_coords = [(stations[sid]['lon'], stations[sid]['lat']) for sid in target_stations]
    for i in range(len(target_coords)):
        for j in range(i+1, len(target_coords)):
            ax1.plot([target_coords[i][0], target_coords[j][0]], 
                    [target_coords[i][1], target_coords[j][1]], 
                    'k--', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('经度 (°)')
    ax1.set_ylabel('纬度 (°)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 目标台站与其他台站的互相关值分布
    ax2.set_title('目标台站与其他台站的互相关值', fontsize=14, fontweight='bold')
    
    # 筛选包含目标台站的数据
    target_related_data = []
    for i, row in cc_data.iterrows():
        if (row['station1'] in target_stations or row['station2'] in target_stations):
            target_related_data.append(row)
    
    if target_related_data:
        target_df = pd.DataFrame(target_related_data)
        
        # 创建台站对标签
        pair_labels = []
        cc_values = []
        for _, row in target_df.iterrows():
            if row['station1'] in target_stations:
                other_station = row['station2']
                target_station = row['station1']
            else:
                other_station = row['station1']
                target_station = row['station2']
            
            label = f"{station_names[target_station]}-{other_station}"
            pair_labels.append(label)
            cc_values.append(row['cc_value'])
        
        # 按互相关值排序
        sorted_data = sorted(zip(pair_labels, cc_values), key=lambda x: x[1], reverse=True)
        sorted_labels, sorted_cc_values = zip(*sorted_data)
        
        # 只显示前10个最高的互相关值
        top_n = min(10, len(sorted_cc_values))
        bars = ax2.bar(range(top_n), sorted_cc_values[:top_n], 
                      color=['skyblue' if '1038' in label else 'lightgreen' if '1812' in label else 'lightcoral' 
                             for label in sorted_labels[:top_n]])
        ax2.set_xticks(range(top_n))
        ax2.set_xticklabels(sorted_labels[:top_n], rotation=45, ha='right')
        ax2.set_ylabel('互相关值')
        ax2.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 3. 距离-时延关系
    ax3.set_title('距离-时延关系', fontsize=14, fontweight='bold')
    
    if target_related_data:
        target_df = pd.DataFrame(target_related_data)
        
        # 为不同目标台站使用不同颜色
        colors = {'001038': 'red', '001812': 'blue', '001827': 'green'}
        for _, row in target_df.iterrows():
            if row['station1'] in target_stations:
                target_station = row['station1']
            else:
                target_station = row['station2']
            
            ax3.scatter(row['distance'], row['timelag'], 
                       s=80, c=colors[target_station], 
                       alpha=0.7, edgecolors='black')
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=colors[sid], markersize=10, 
                                     label=f'台站{station_names[sid]}') 
                          for sid in target_stations]
        ax3.legend(handles=legend_elements)
        
        ax3.set_xlabel('距离 (km)')
        ax3.set_ylabel('时延 (s)')
        ax3.grid(True, alpha=0.3)
    
    # 4. 互相关值-距离关系
    ax4.set_title('互相关值-距离关系', fontsize=14, fontweight='bold')
    
    if target_related_data:
        target_df = pd.DataFrame(target_related_data)
        
        # 为不同目标台站使用不同颜色
        for _, row in target_df.iterrows():
            if row['station1'] in target_stations:
                target_station = row['station1']
            else:
                target_station = row['station2']
            
            ax4.scatter(row['distance'], row['cc_value'], 
                       s=80, c=colors[target_station], 
                       alpha=0.7, edgecolors='black')
        
        ax4.set_xlabel('距离 (km)')
        ax4.set_ylabel('互相关值')
        ax4.legend(handles=legend_elements)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('three_stations_cc_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细信息
    print("=== 台站1827、1812、1038互相关分析结果 ===")
    print(f"台站信息:")
    for sid in target_stations:
        info = stations[sid]
        print(f"  台站{station_names[sid]}: 纬度={info['lat']:.5f}°, 经度={info['lon']:.5f}°, 高程={info['elevation']:.1f}m")
    
    print(f"\n目标台站与其他台站的互相关结果 (按互相关值排序):")
    if target_related_data:
        target_df = pd.DataFrame(target_related_data)
        
        # 创建更易读的标签
        readable_pairs = []
        for _, row in target_df.iterrows():
            if row['station1'] in target_stations:
                other_station = row['station2']
                target_station = row['station1']
            else:
                other_station = row['station1']
                target_station = row['station2']
            
            readable_pairs.append({
                'pair': f"{station_names[target_station]}-{other_station}",
                'distance': row['distance'],
                'timelag': row['timelag'],
                'cc_value': row['cc_value']
            })
        
        # 按互相关值排序
        readable_pairs.sort(key=lambda x: x['cc_value'], reverse=True)
        
        for pair in readable_pairs[:10]:  # 显示前10个
            print(f"  {pair['pair']}: 距离={pair['distance']:.2f}km, 时延={pair['timelag']:.1f}s, 互相关值={pair['cc_value']:.6f}")
    
    print(f"\n注意: 数据中没有1827-1812、1827-1038、1812-1038这三个台站对之间的直接互相关数据。")
    print(f"以上显示的是这三个台站与其他台站的互相关结果。")

if __name__ == "__main__":
    plot_three_stations_cc() 