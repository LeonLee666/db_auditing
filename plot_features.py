import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import argparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from feature_engineering import extract_and_parse_sql_file

def plot_distribution(values_df, outfile, filter_w_id=None, y_limits=None, title_prefix=''):
        """
        绘制s_w_id和s_i_id的分布图
        
        参数:
        - values_df: 输入数据DataFrame
        - outfile: 输出文件名
        - filter_w_id: 可选，用于筛选特定s_w_id的值
        - y_limits: 可选，格式为[(y1_min, y1_max), (y2_min, y2_max)]，用于设置y轴范围
        - title_prefix: 可选，标题前缀
        """
        # 数据筛选
        if filter_w_id is not None:
            df = values_df[values_df['s_w_id'] == filter_w_id].copy()
            w_id_desc = f'where s_w_id = {filter_w_id}'
        else:
            df = values_df.copy()
            w_id_desc = '(all values)'
            
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
        
        # 准备数据
        df = df.reset_index(drop=True)
        print(f"数据总行数: {len(df)}")
        start_idx = 100000
        end_idx = 100100
        df_subset = df.iloc[start_idx:end_idx]
        print(f"选取的数据行数: {len(df_subset)}")
        
        # 绘制上面的子图 (s_w_id)
        ax1.plot(df_subset.index, df_subset['s_w_id'],
                color='black', marker='o', linestyle='-', label='s_w_id')
        ax1.set_title(f'{title_prefix}Distribution of s_w_id {w_id_desc}', fontsize=14)
        ax1.set_xlabel('Row Number', fontsize=14)
        ax1.set_ylabel('s_w_id', fontsize=14)
        ax1.tick_params(labelsize=14, axis='x', rotation=15)
        ax1.grid(True)
        ax1.legend(fontsize=14)
        
        # 绘制下面的子图 (s_i_id)
        ax2.plot(df_subset.index, df_subset['s_i_id'],
                color='black', marker='o', linestyle='-', label='s_i_id')
        ax2.set_title(f'{title_prefix}Distribution of s_i_id {w_id_desc}', fontsize=14)
        ax2.set_xlabel('Row Number', fontsize=14)
        ax2.set_ylabel('s_i_id', fontsize=14)
        ax2.tick_params(labelsize=14, axis='x', rotation=15)
        ax2.grid(True)
        ax2.legend(fontsize=14)
        
        # 设置y轴范围（如果提供）
        if y_limits:
            ax1.set_ylim(*y_limits[0])
            ax2.set_ylim(*y_limits[1])
        
        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f'fig/{outfile}', format='pdf', bbox_inches='tight')
        plt.close()

def plot_2d_20points(infile, outfile='2d.pdf'):
    values_df = extract_and_parse_sql_file(infile)
    values_df = values_df.rename(columns={'value_0': 's_w_id', 'value_1': 's_i_id'})
    
    # 调用新函数处理特定范围的数据
    filtered_df = plot_filtered_range(values_df, 'temp.pdf')
    
    # 绘制原始的s_w_id=5分布图
    plot_distribution(values_df, 's_w_id_5.pdf', filter_w_id=5)
    
    # 绘制放大的s_w_id=5分布图
    plot_distribution(values_df, 's_w_id_5_zoomed.pdf', 
                     filter_w_id=5,
                     y_limits=[(4.9, 5.1), (60560, 60620)],
                     title_prefix='Zoomed ')
    
    # 绘制所有数据的分布图
    plot_distribution(values_df, 'all_w_id.pdf')
    
    scaler = MinMaxScaler()
    values_scaled = pd.DataFrame(
        scaler.fit_transform(values_df),
        columns=values_df.columns
    )
    # 定义数据点的范围
    start_idx = 1000000
    end_idx = 1000100
    
    # 创建主图
    fig, ax = plt.subplots(figsize=(5, 5))
    # 绘制主图内容
    ax.scatter(values_scaled['s_w_id'][start_idx:end_idx], values_scaled['s_i_id'][start_idx:end_idx], 
              c='black', marker='o', label=f'Points {start_idx}-{end_idx}')
    # 创建放大的图，调整大小并移到右上角
    axins = inset_axes(ax, width=0.7, height=0.7,  # 减小子图大小
                      bbox_to_anchor=(0.95, 0.95),  # 修改为右上角位置
                      bbox_transform=ax.transAxes,
                      loc='upper right')  # 修改为右上角
    # 在子图中绘制相同的数据
    axins.scatter(values_scaled['s_w_id'][start_idx:end_idx], values_scaled['s_i_id'][start_idx:end_idx], 
                 c='black', marker='o')
    # 设置子图的范围和样式
    axins.set_xlim(0.72, 0.82)
    axins.set_ylim(0.962, 0.96215)
    axins.grid(True)
    # 隐藏刻度值，只保留刻度线
    axins.set_xticks([])
    axins.set_yticks([])
    # 为放大区域的点添加标注
    for idx, i in enumerate(range(start_idx, end_idx), 1):
        axins.annotate(r'$L(s_{i_{%d}})$' % idx, 
                      (values_scaled['s_w_id'][i], values_scaled['s_i_id'][i]),
                      xytext=(10, 0),  # 水平偏移10个点，垂直不偏移
                      textcoords='offset points',
                      verticalalignment='center',  # 垂直居中对齐
                      fontsize=8)
    # 在主图上添加蓝色方框标记
    ax.plot(0.75, 0.962, 's', color='blue', markersize=8)
    
    # 设置主图的其他属性
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.set_xlabel('s_w_id')
    ax.set_ylabel('s_i_id')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(True)
    ax.legend(loc='upper left')
    
    # 使用annotate添加带箭头的蓝色连接线
    ax.annotate('', 
               xy=(0.85, 0.9),
               xytext=(0.75, 0.962),
               arrowprops=dict(arrowstyle='->', color='blue', linewidth=1))
    
    # 为子图添加蓝色边框
    for spine in axins.spines.values():
        spine.set_edgecolor('blue')
        spine.set_linestyle((0, (2, 4)))
    
    plt.savefig(f'fig/{outfile}', format='pdf', bbox_inches='tight')
    plt.close()

def plot_filtered_range(values_df, outfile='temp.pdf'):
    """
    获取指定范围的数据，过滤s_w_id=5的记录并绘制分布图
    
    参数:
    - values_df: 输入数据DataFrame
    - outfile: 输出文件名，默认为'temp.pdf'
    """
    # 获取指定范围的数据
    start_idx = 1000000
    end_idx = 1001000
    df_range = values_df.iloc[start_idx:end_idx].copy()
    print(f"选取的范围数据总行数: {len(df_range)}")
    
    # 过滤s_w_id=5的数据
    df_filtered = df_range[df_range['s_w_id'] == 4].copy()
    print(f"过滤后的数据行数: {len(df_filtered)}")
    
    # 重置索引用于绘图
    df_filtered = df_filtered.reset_index(drop=True)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 绘制上面的子图 (s_w_id)
    ax1.plot(df_filtered.index, df_filtered['s_w_id'],
            color='black', marker='o', linestyle='-', label='s_w_id')
    ax1.set_title('Distribution of s_w_id = 4 (range 1000000-1001000)', fontsize=14)
    ax1.set_xlabel('Row Number', fontsize=14)
    ax1.set_ylabel('s_w_id', fontsize=14)
    ax1.tick_params(labelsize=14)
    ax1.grid(True)
    ax1.legend(fontsize=14)
    
    # 绘制下面的子图 (s_i_id)
    ax2.plot(df_filtered.index, df_filtered['s_i_id'],
            color='black', marker='o', linestyle='-', label='s_i_id')
    ax2.set_title('Distribution of s_i_id where s_w_id = 4 (range 1000000-1001000)', fontsize=14)
    ax2.set_xlabel('Row Number', fontsize=14)
    ax2.set_ylabel('s_i_id', fontsize=14)
    ax2.tick_params(labelsize=14)
    ax2.set_ylim(95000, 100000)
    ax2.grid(True)
    ax2.legend(fontsize=14)
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f'fig/{outfile}', format='pdf', bbox_inches='tight')
    plt.close()
    
    return df_filtered  # 返回过滤后的数据，以便需要时进行进一步分析

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot points with zoomed inset')
    parser.add_argument('--infile', type=str, help='Input file path')
    parser.add_argument('--outfile', type=str, default='2d.pdf', 
                       help='Output image path (default: 2d.pdf)')
    args = parser.parse_args()
    plot_2d_20points(args.infile, args.outfile) 