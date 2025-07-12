import matplotlib.pyplot as plt
import pandas as pd

def plot_points_from_file(file_path):
    try:
        # 读取文件，假设文件以空格或逗号分隔
        df = pd.read_csv(file_path, delim_whitespace=True, header=None, 
                         names=['号码', '纬度', '经度', '海拔'])
        
        # 提取经纬度数据
        latitudes = df['纬度']
        longitudes = df['经度']
        numbers = df['号码']
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制原始点
        plt.scatter(longitudes, latitudes, color='red', s=50, alpha=0.7, label='原始数据点')
        
        #为每个点添加号码标注
        for lat, lon, num in zip(latitudes, longitudes, numbers):
            plt.annotate(str(num), (lon, lat), textcoords="offset points", 
                         xytext=(0,10), ha='center', fontsize=9)
        
        # 添加指定的蓝色点 (20, 110.3)
        special_lat, special_lon = 20, 110.3
        plt.scatter(special_lon, special_lat, color='blue', s=80, alpha=0.9, label='特殊点 (20, 110.3)')
        plt.annotate("typhone", (special_lon, special_lat), textcoords="offset points", 
                     xytext=(0,15), ha='center', fontsize=10, color='blue')
        
        # 找出指定号码的点并连接
        target_numbers = [1827, 1812, 1038]  # 使用完整字符串格式的号码
        target_points = []
        
        for num in target_numbers:
            point = df[df['号码'] == num]
            if not point.empty:
                target_points.append((point.iloc[0]['经度'], point.iloc[0]['纬度'], num))
                print(f"找到点 {num}: 经度={point.iloc[0]['经度']}, 纬度={point.iloc[0]['纬度']}")
            else:
                print(f"警告：未找到号码为 {num} 的点")
        
        # 连接三个点（绘制无限延伸的直线）
        if len(target_points) >= 2:
            # 获取当前坐标轴范围
            x_min, x_max = plt.xlim()
            y_min, y_max = plt.ylim()
            
            # 为不同的连线创建不同的偏移量，避免标注重叠
            offset_map = {}
            for i in range(len(target_points)):
                for j in range(i+1, len(target_points)):
                    # 创建连线的唯一标识
                    line_id = f"{min(target_points[i][2], target_points[j][2])}-{max(target_points[i][2], target_points[j][2])}"
                    # 为不同的连线分配不同的角度偏移
                    if len(target_points) == 3:
                        # 三个点形成三条连线的情况，均匀分配角度
                        angles = [30, 90, 150]
                        offset_map[line_id] = angles[len(offset_map) % 3]
                    else:
                        # 更多点的情况，使用默认角度
                        offset_map[line_id] = 30 * (len(offset_map) % 12)
            
            for i in range(len(target_points)):
                for j in range(i+1, len(target_points)):
                    x1, y1, num1 = target_points[i]
                    x2, y2, num2 = target_points[j]
                    
                    # 创建连线的唯一标识
                    line_id = f"{min(num1, num2)}-{max(num1, num2)}"
                    angle = offset_map[line_id]
                    
                    # 计算直线方程 y = kx + b
                    if x2 != x1:
                        k = (y2 - y1) / (x2 - x1)
                        b = y1 - k * x1
                        
                        # 计算直线在当前坐标轴范围内的端点
                        x_line = [x_min, x_max]
                        y_line = [k * x + b for x in x_line]
                        
                        # 绘制直线
                        plt.plot(x_line, y_line, 'g-', linewidth=1.5, alpha=0.7)
                        
                        # 计算直线中点并添加标注
                        mid_x = (x1 + x2) / 2
                        mid_y = (y1 + y2) / 2
                        
                        # 使用角度计算偏移量，避免标注重叠
                        distance = 20  # 标注与线的距离
                        dx = distance * pd.np.cos(pd.np.radians(angle))
                        dy = distance * pd.np.sin(pd.np.radians(angle))
                        
                        plt.annotate(f"{num1}-{num2}", (mid_x, mid_y), textcoords="offset points", 
                                    xytext=(dx, dy), ha='center', fontsize=9, color='green',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
                    else:
                        # 垂直直线的情况
                        plt.axvline(x=x1, color='g', linewidth=1.5, alpha=0.7)
                        
                        # 在垂直直线中点添加标注
                        mid_y = (y1 + y2) / 2
                        
                        # 使用角度计算偏移量，避免标注重叠
                        distance = 20  # 标注与线的距离
                        dx = distance * pd.np.cos(pd.np.radians(angle))
                        dy = distance * pd.np.sin(pd.np.radians(angle))
                        
                        plt.annotate(f"{num1}-{num2}", (x1, mid_y), textcoords="offset points", 
                                    xytext=(dx, dy), ha='center', fontsize=9, color='green',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
            
            print(f"成功绘制 {len(target_points)} 个点之间的无限延伸直线")
        
        # 添加标题和标签
        plt.title('数据点分布图')
        plt.xlabel('lon')
        plt.ylabel('lat')
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置坐标轴范围，确保包含所有点和特殊点
        all_lons = list(longitudes) + [special_lon]
        all_lats = list(latitudes) + [special_lat]
        plt.xlim(min(all_lons) - 5, max(all_lons) + 5)
        plt.ylim(min(all_lats) - 5, max(all_lats) + 5)
        
        # 添加图例
        plt.legend()
        
        # 显示图形
        plt.tight_layout()
        plt.show()
        
        print(f"成功绘制 {len(df)} 个原始数据点和 1 个特殊点")
        
    except FileNotFoundError:
        print(f"错误：未找到文件 '{file_path}'")
    except Exception as e:
        print(f"发生未知错误：{e}")

if __name__ == "__main__":
    file_path = "chuanxi_info.txt"  # 请替换为实际文件路径
    plot_points_from_file(file_path)