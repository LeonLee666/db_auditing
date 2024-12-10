import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import ast
import argparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def plot_first_20_points(infile, outfile='first_20_points.png'):
    
    # 数据处理部分保持不变
    df = pd.read_csv(infile)
    literals_parsed = []
    for row in df['literals']:
        try:
            parsed = ast.literal_eval(row)
            literals_parsed.append(parsed)
        except (ValueError, SyntaxError) as e:
            print(f'解析错误: {e}')
            literals_parsed.append([])
            
    values_df = pd.DataFrame(literals_parsed, columns=['s_w_id', 's_i_id'])
    scaler = MinMaxScaler()
    values_scaled = pd.DataFrame(
        scaler.fit_transform(values_df),
        columns=values_df.columns
    )
    
    # 创建主图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制主图内容
    ax.scatter(values_scaled['s_w_id'][2500000:2500050], values_scaled['s_i_id'][2500000:2500050], 
              c='blue', marker='o', label='Points 2500000-2500050')
    
    # 创建放大的���图
    axins = inset_axes(ax, width=2.5, height=2.5,
                      bbox_to_anchor=(0.95, 0.05),
                      bbox_transform=ax.transAxes,
                      loc='lower right')
    
    # 在子图中绘制相同的数据
    axins.scatter(values_scaled['s_w_id'][2500000:2500050], values_scaled['s_i_id'][2500000:2500050], 
                 c='blue', marker='o')
    
    # 设置子图的范围和样式
    axins.set_xlim(0.4, 0.6)
    axins.set_ylim(0.70874, 0.70880)
    axins.grid(True)
    
    # 为子图添加边框
    for spine in axins.spines.values():
        spine.set_edgecolor('red')
        spine.set_linestyle('--')
    
    # 为放大区域的点添加标注
    for idx, i in enumerate(range(2500000, 2500050), 1):
        axins.annotate(r'$L(s_{i_%d})$' % idx, 
                      (values_scaled['s_w_id'][i], values_scaled['s_i_id'][i]),
                      xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 设置主图的其他属性
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Distribution of Points')
    ax.set_xlabel('s_w_id')
    ax.set_ylabel('s_i_id')
    ax.grid(True)
    ax.legend()
    
    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot points with zoomed inset')
    parser.add_argument('--infile', type=str, help='Input file path')
    parser.add_argument('--outfile', type=str, default='points_with_inset.png', 
                       help='Output image path (default: points_with_inset.png)')
    
    args = parser.parse_args()
    plot_first_20_points(args.infile, args.outfile) 