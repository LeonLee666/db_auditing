import pandas as pd
import matplotlib.pyplot as plt


def setup_plot_params():
    """设置全局绘图参数"""
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14


def plot_metrics(df):
    """绘制训练指标图表"""
    # 绘制准确率图
    plt.figure(figsize=(6, 4))
    plt.grid(True)
    plt.plot(df['step'], df['fit accuracy'], 'k-', label='Training Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy (First 3000 Steps)')
    plt.ylim(0, 1)  # 设置y轴范围
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig/vanilla-accuracy.pdf')
    plt.close()

    # 绘制损失图
    plt.figure(figsize=(6, 4))
    plt.grid(True)
    plt.plot(df['step'], df['fit loss'], 'k-', label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss (First 3000 Steps)')
    plt.ylim(0, 1)  # 设置y轴范围
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig/vanilla-loss.pdf')
    plt.close()


def main():
    """主函数"""
    # 读取CSV日志文件并筛选前3000步的数据
    df = pd.read_csv('metrics/audit/version_0/metrics.csv')
    df = df[df['step'] <= 3000]
    
    # 设置绘图参数
    setup_plot_params()
    
    # 绘制图表
    plot_metrics(df)


if __name__ == '__main__':
    main() 