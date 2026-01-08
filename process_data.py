import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def var3design(n):
    d = pow(2, n)
    return 2 * (d - 1) / (d + 2)

def varRC(n):
    d = pow(2, n)
    return (d + 1) * (1 + 1 / d) -  4

def analyze_distribution(filepath, save_plot=True):
    """
    分析npy文件中数据的分布情况
    
    参数:
        filepath: npy文件的路径
        save_plot: 是否保存图表
    """
    data = np.load(filepath)
    
    print("=" * 70)
    print(f"分析文件: {filepath}")
    print("=" * 70)
    print("\n数据基本统计信息")
    print("-" * 70)
    print(f"数据样本数: {len(data)}")
    print(f"数据类型: {data.dtype}")
    print(f"数据范围: [{np.min(data):.6e}, {np.max(data):.6e}]")
    print(f"平均值: {np.mean(data):.6e}")
    print(f"标准差: {np.std(data):.6e}")
    print(f"方差: {np.var(data):.6e}")
    print(f"中位数: {np.median(data):.6e}")
    print(f"四分位数: Q1={np.percentile(data, 25):.6e}, Q3={np.percentile(data, 75):.6e}")
    print(f"偏度: {stats.skew(data):.6f}")
    print(f"峰度: {stats.kurtosis(data):.6f}")

    print("\n" + "-" * 70)
    print("数据分布信息")
    print("-" * 70)

    # 查看数据中的唯一值和频率
    unique_values, counts = np.unique(data, return_counts=True)
    print(f"唯一值个数: {len(unique_values)}")
    print(f"\n最常见的10个值及其频率:")
    sorted_indices = np.argsort(-counts)[:10]
    for idx in sorted_indices:
        print(f"  {unique_values[idx]:.6e}: 出现 {counts[idx]:6d} 次 ({counts[idx]/len(data)*100:6.2f}%)")

    # 统计负值、零值、正值
    neg_count = np.sum(data < 0)
    zero_count = np.sum(data == 0)
    pos_count = np.sum(data > 0)
    print(f"\n负数个数: {neg_count} ({neg_count/len(data)*100:.2f}%)")
    print(f"零值个数: {zero_count} ({zero_count/len(data)*100:.2f}%)")
    print(f"正数个数: {pos_count} ({pos_count/len(data)*100:.2f}%)")

    # 正态性检验
    _, p_value = stats.normaltest(data)
    print(f"\n正态性检验 p-value: {p_value:.6f}")
    if p_value < 0.05:
        print("数据不服从正态分布 (p < 0.05)")
    else:
        print("数据可能服从正态分布 (p >= 0.05)")

    # 绘制图表
    if save_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 将离散值映射到等距刻度，避免横轴因取值跨度过大而显得稀疏
        x_pos = np.arange(len(unique_values))
        xtick_labels = [f"{v:.3g}" for v in unique_values]

        # 1. 离散值柱状图（每个唯一值单独一根柱）
        axes[0, 0].bar(x_pos, counts, width=0.8, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(xtick_labels, rotation=45, ha='right')
        axes[0, 0].set_xlabel('Value (discrete)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Histogram of Data (per value)', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 对数柱状图（同样按唯一值分桶）
        axes[0, 1].bar(x_pos, counts, width=0.8, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(xtick_labels, rotation=45, ha='right')
        axes[0, 1].set_xlabel('Value (discrete)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency (log scale)', fontsize=12)
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('Histogram (Log Scale, per value)', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 累积分布函数 (CDF)
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 0].plot(sorted_data, cdf, linewidth=2, color='green')
        axes[1, 0].set_xlabel('Value', fontsize=12)
        axes[1, 0].set_ylabel('Cumulative Probability', fontsize=12)
        axes[1, 0].set_title('Cumulative Distribution Function (CDF)', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 箱线图
        axes[1, 1].boxplot(data, vert=True)
        axes[1, 1].set_ylabel('Value', fontsize=12)
        axes[1, 1].set_title('Box Plot', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        
        # 生成输出文件名
        output_path = f'./data/Hadamard/distribution_d{2}_n{14}_m{1}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存到 {output_path}")
        plt.close()

    print("=" * 70 + "\n")
    
    return data


if __name__ == "__main__":

    # 分析单个文件的分布
    # analyze_distribution('./data/Hadamard/d=2,n=14,m=1.npy')

    # multilayer
    # n = np.arange(3, 10)
    # Y = []
    # for m in [1]:
    #     Ym = []
    #     for i in n:
    #         data = np.load(f'./data/Hadamard/d=2,n={i},m={m}.npy')
    #         var = np.var(data)
    #         Ym.append(var)
    #     Y.append(Ym)
    # fig = plt.figure(figsize=(8,6))
    # for m in [1]:
    #     plt.plot(n,Y[m-1], marker='o', markersize=5, label=f'Numerical')
    # plt.plot(n, var3design(n), marker='s', markersize=5, label='3-design variance')
    # exact_h = {2: 0.500150, 3: 0.932124, 4: 1.672027, 5: 2.587864, 6: 3.714565, 7: 4.732950, 8: 6.054775, 9: 6.642699}
    # exact = var3design(n) + np.array([exact_h[i] for i in n])
    # plt.plot(n, exact, marker='^', markersize=5, label='Theoretical')
    # plt.xlabel(r'$n$')
    # plt.ylabel('Variance')
    # # plt.title('d=2,n=50,N=20000,multilayer')
    # plt.legend()
    # plt.show()

    # # XYZ
    n = np.arange(2, 10)
    Y = []
    for i in n:
        if i == 4:
            data = np.load(f'./data/XYZ/d=2,n={i},sample=50000,XYZ.npy')
        else:
            data = np.load(f'./data/XYZ/d=2,n={i},XYZ.npy')
        var = np.var(data)
        Y.append(var)
    fig = plt.figure(figsize=(8,6))
    n_1 = np.arange(3, 11)
    exact_lc = {2: -0.000363, 3: -0.001063, 4: 0.111268, 5: 0.147949, 6: 0.147336, 7: 0.059895, 8: 0.057927, 9: 0.081064}
    exact = var3design(n) + np.array([exact_lc[i] for i in n])
    plt.plot(n,Y, marker='o', markersize=5, label='Numerical')
    plt.plot(n, var3design(n), marker='s', markersize=5, label='3-design variance')
    plt.plot(n, exact, marker='^', markersize=5, label='Theoretical')
    plt.legend()
    # plt.plot(n_1, UB, marker='^', markersize=5)

    plt.xlabel(r'$n$')
    plt.ylabel('Variance')
    plt.show()

    # X = [1000*i for i in range(1,21)]
    # Y,Z = [],[]
    # i = 10
    # data = np.load(f'0325/XYZ/d=2,n=10,XYZ,rand{i}.npy')
    # for N in X:
    #     tmp = np.random.choice(data, N, replace=False)
    #     mean = np.mean(tmp)
    #     Y.append(mean)
    #     var = np.var(tmp)
    #     Z.append(var)
    # fig = plt.figure(figsize=(8,6))
    # ax1 = fig.add_subplot(111)
    # ax1.plot(X,Y, 'o-')
    # ax1.set_xlabel('Sample Number')
    # ax1.set_ylabel('Mean')
    # ax1.set_xticks([2000*i for i in range(1,11)])
    # ax2 = ax1.twinx()
    # ax2.plot(X,Z, 's--', color='orange')
    # ax2.set_ylabel('Variance')
    # ax2.set_title('d=2,n=10,XYZ')
    # plt.show()

    # singlelayer
    # X = np.arange(11)
    # Y = []
    # for j in range(10):
    #     Yj = []
    #     for h in X:
    #         data = np.load(f'0325/singlelayer/d=2,n=10,h={h},XYZ,rand{j}.npy')
    #         var = np.var(data)
    #         Yj.append(var)
    #     Y.append(Yj)
    # fig = plt.figure(figsize=(8,6))
    # for j in range(10):
    #     plt.plot(X,Y[j],'o:')
    # # plt.yscale('log')
    # plt.xlabel('Number of H')
    # plt.ylabel('Variance')
    # plt.title('d=2,n=10,N=50000,singlelayer')
    # plt.show()
