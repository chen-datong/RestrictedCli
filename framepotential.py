import trace
from mpi4py import MPI
import numpy as np
from randomSL import *
from magic_state import *
from main2 import restricted_Clifford_update
from tqdm import tqdm
import matplotlib.pyplot as plt

def restricted_Clifford_update_noid(stabVecs,finite_field,n):
    stabZ, stabX = stabVecs[:,:n], stabVecs[:,n:2*n]
    ZvList = [list(stabZ[i]) for i in range(len(stabZ))]
    XvList = [list(stabX[i]) for i in range(len(stabX))]
    new_XvList, new_ZvList = random_SL_transformation_noidentity(finite_field,XvList,ZvList)
    new_stabX, new_stabZ = [], []
    for i in range(len(new_XvList)):
        new_stabX.append(np.array(new_XvList[i],dtype=int))
        new_stabZ.append(np.array(new_ZvList[i],dtype=int))
    new_stabX, new_stabZ = np.array(new_stabX,dtype=int), np.array(new_stabZ,dtype=int)
    k = stabVecs.shape[0]
    new_stabVecs = np.zeros((k,2*n), dtype=int)
    new_stabVecs[:,:n], new_stabVecs[:,n:2*n] = new_stabZ, new_stabX
    return new_stabVecs

def Haar_3rdFP(n):
    d = pow(2, n)
    return 6 / (d + 2) / (d + 1) / d

def zero(d,n):
    stabVecs = np.concatenate([np.identity(n,dtype=int),np.zeros((n,n),dtype=int)], axis=1)
    phaseVec = np.zeros(n,dtype=int)
    return stabVecs, phaseVec

def single_LC_3rdFP(n, layer, finite_field):
    """
    n: qubit number
    layer: interleaved layer of the Local Clifford group
    """
    stabVecs, phaseVec = zero(2,n)
    new_stabVecs = restricted_Clifford_update(stabVecs,finite_field,n,n)
    rho = StabState2(n, new_stabVecs,phaseVec)
    a = np.random.randint(2, size=2*n, dtype=int)
    rho.phase_update(a)
    for _ in range(2 * layer):
        flag = np.random.randint(6,size=n)
        for k in range(n):
            if flag[k]==1:
                rho.Hadamard(k)
            elif flag[k]==2:
                rho.Phase(k)
            elif flag[k]==3:
                rho.Phase(k)
                rho.Hadamard(k)
            elif flag[k]==4:
                rho.Hadamard(k)
                rho.Phase(k)
            elif flag[k]==5:
                rho.Hadamard(k)
                rho.Phase(k)
                rho.Hadamard(k)            
        new_stabVecs = restricted_Clifford_update(rho.stabVecs.copy(),finite_field,n,n)
        rho.stabVecs = new_stabVecs
        a = np.random.randint(2, size=2*n, dtype=int)
        rho.phase_update(a)

    p = rho.inner_zero()
    return pow(p, 3)

def experiment_LC(n_start, n_end, layer, samples):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if rank == 0:
        fp = []
    
    for n in range(n_start, n_end+1):
        finite_field = FiniteField(2, n)
        np.random.seed(rank + n)
        # 把 samples 尽量平均分到各个 rank（处理不能整除的情况）
        local_samples = samples // size + (1 if rank < (samples % size) else 0)

        local_sum = 0.0
        if rank == 0:
            for _ in range(local_samples):
                local_sum += single_LC_3rdFP(n, layer, finite_field)
        else:
            for _ in range(local_samples):
                local_sum += single_LC_3rdFP(n, layer, finite_field)
        
        all_result = comm.gather(local_sum, root=0)
        if rank == 0:
            all_result = np.array(all_result)
            sum_result = np.sum(all_result)
            sum_result /= samples
            haar_fp = Haar_3rdFP(n)
            result = sum_result / haar_fp - 1
            fp.append(result)
            print(f"n={n} done, result={result}")
    
    if rank == 0:
        np.save(f'./data/LC_3rdFP_{n_start}To{n_end}_layer{layer}_samples{samples}.npy', np.array(fp))
        print(fp)

def single_H_3rdFP(n, layer, finite_field):
    """
    n: qubit number
    layer: interleaved layer of Hadamard gates
    """
    stabVecs, phaseVec = zero(2,n)
    new_stabVecs = restricted_Clifford_update(stabVecs,finite_field,n,n)
    rho = StabState2(n,new_stabVecs,phaseVec)
    a = np.random.randint(2, size=2*n, dtype=int)
    rho.phase_update(a)
    for _ in range(2 * layer):
        flag = np.random.randint(2,size=n)
        for k in range(n):
            if flag[k]: rho.Hadamard(k)
        new_stabVecs = restricted_Clifford_update(rho.stabVecs,finite_field,n,n)
        rho.stabVecs = new_stabVecs
        a = np.random.randint(2, size=2*n, dtype=int)
        rho.phase_update(a)
    p = rho.inner_zero()
    return pow(p, 3)

def experiment_H(n_start, n_end, layer, samples):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if rank == 0:
        fp = []
    
    for n in range(n_start, n_end+1):
        finite_field = FiniteField(2, n)
        np.random.seed(rank + n)
        # 把 samples 尽量平均分到各个 rank（处理不能整除的情况）
        local_samples = samples // size + (1 if rank < (samples % size) else 0)

        local_sum = 0.0
        if rank == 0:
            for _ in range(local_samples):
                local_sum += single_H_3rdFP(n, layer, finite_field)
        else:
            for _ in range(local_samples):
                local_sum += single_H_3rdFP(n, layer, finite_field)
        
        all_result = comm.gather(local_sum, root=0)
        if rank == 0:
            all_result = np.array(all_result)
            sum_result = np.sum(all_result)
            sum_result /= samples
            haar_fp = Haar_3rdFP(n)
            result = sum_result / haar_fp - 1
            fp.append(result)
            print(f"n={n} done, result={result}")
    
    if rank == 0:
        np.save(f'./data/Hadamard_3rdFP_{n_start}To{n_end}_layer{layer}_samples{samples}.npy', np.array(fp))
        print(fp)


def MaxEntangle(state, n):
    for i in range(n):
        state.Hadamard(i)
    for i in range(n):
        state.CNOT(i, i+n)

def MaxEntangle_inverse(state, n):
    for i in range(n):
        state.CNOT(i, i+n)
    for i in range(n):
        state.Hadamard(i)

def apply_restricted_clifford_on_first_register(state, finite_field, n, with_identity=True):
    """
    Apply U ⊗ I, where U is defined by restricted_Clifford_update.
    """
    stab_first = np.concatenate(
        [state.stabVecs[:, 0:n], state.stabVecs[:, 2*n:2*n+n]],
        axis=1
    )  # shape: (2n, 2n)

    # Apply your restricted Clifford update (acts on n-qubit labels [Z|X])
    if with_identity:
        new_stab_first = restricted_Clifford_update(stab_first, finite_field, n, n)  # (2n,2n)
    else:
        new_stab_first = restricted_Clifford_update_noid(stab_first, finite_field, n)  # (2n,2n)

    # Write back to the big tableau
    state.stabVecs[:, 0:n] = new_stab_first[:, 0:n]                 # Z_first
    state.stabVecs[:, 2*n:2*n+n] = new_stab_first[:, n:2*n]         # X_first
    if with_identity:
        a = np.random.randint(2, size=2*n, dtype=int)
    else:
        while True:
            a = np.random.randint(2, size=2*n, dtype=int)
            if np.any(a):
                break
    new_a = np.zeros(4*n, dtype=int)
    new_a[0:n] = a[0:n]
    new_a[2*n:2*n+n] = a[n:2*n]
    state.phase_update(new_a)



def traceU(finite_field, n, layer, type='LC', with_identity=True):
    """
    Returns |Tr(U)|^2 where U is drawn via your restricted_Clifford_update.
    """
    stabVecs, phaseVec = zero(2, 2*n)        # |0^{2n}>
    st = StabState2(2*n, stabVecs, phaseVec)
    MaxEntangle(st, n)       # -> |Phi>
    apply_restricted_clifford_on_first_register(st, finite_field, n, with_identity=with_identity)  # U ⊗ I
    for _ in range(2 * layer):
        if type=='LC':
            flag = np.random.randint(6,size=n)
            for k in range(n):
                if flag[k]==1:
                    st.Hadamard(k)
                elif flag[k]==2:
                    st.Phase(k)
                elif flag[k]==3:
                    st.Phase(k)
                    st.Hadamard(k)
                elif flag[k]==4:
                    st.Hadamard(k)
                    st.Phase(k)
                elif flag[k]==5:
                    st.Hadamard(k)
                    st.Phase(k)
                    st.Hadamard(k)            
        elif type=='H':
            flag = np.random.randint(2,size=n)
            for k in range(n):
                if flag[k]: st.Hadamard(k)
        apply_restricted_clifford_on_first_register(st, finite_field, n, with_identity=with_identity)  # U ⊗ I
    MaxEntangle_inverse(st, n)            # back
    p = st.inner_zero()              # = |<0|W|0>|^2
    return p

def collect_p_distribution(n, layer, samples, type='LC'):
    """
    收集p值的分布统计
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    finite_field = FiniteField(2, n)
    np.random.seed(rank + n)
    local_samples = samples // size + (1 if rank < (samples % size) else 0)
    
    local_p_values = []
    if rank == 0:
        for _ in tqdm(range(local_samples), desc=f"Collecting p values for n={n}"):
            p = traceU(finite_field, n, layer, type=type) * (4 ** n)
            local_p_values.append(p)
    else:
        for _ in range(local_samples):
            p = traceU(finite_field, n, layer, type=type) * (4 ** n)
            local_p_values.append(p)
    
    # 收集所有进程的p值
    all_p_values = comm.gather(local_p_values, root=0)
    
    if rank == 0:
        # 合并所有p值
        all_p_values = np.concatenate(all_p_values)
        
        # 统计分布
        unique_values, counts = np.unique(all_p_values, return_counts=True)
        probabilities = counts / len(all_p_values)
        
        print(f"\n=== P值分布统计 (n={n}, layer={layer}, samples={samples}) ===")
        print(f"共有 {len(unique_values)} 个不同的p值")
        print(f"p值范围: [{np.min(all_p_values):.6e}, {np.max(all_p_values):.6e}]")
        print(f"平均值: {np.mean(all_p_values):.6e}")
        print(f"标准差: {np.std(all_p_values):.6e}")
        print("\n前10个最常见的p值及其概率:")
        sorted_indices = np.argsort(-counts)[:10]
        for i in sorted_indices:
            print(f"  p = {unique_values[i]:.6e}, 出现概率 = {probabilities[i]:.4f} ({counts[i]}次)")
        
        # # 保存详细数据
        # np.savez(f'./data/{type}_p_distribution_n{n}_layer{layer}_samples{samples}.npz',
        #          all_p_values=all_p_values,
        #          unique_values=unique_values,
        #          counts=counts,
        #          probabilities=probabilities)
        # print(f"\n分布数据已保存到 ./data/{type}_p_distribution_n{n}_layer{layer}_samples{samples}.npz")
        
        # 画图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 直方图 - 显示p值的连续分布
        axes[0, 0].hist(all_p_values, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('p value', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title(f'Histogram of p values (n={n}, layer={layer})', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 概率分布 - 显示最常见的p值
        top_n = min(20, len(unique_values))
        top_indices = np.argsort(-counts)[:top_n]
        axes[0, 1].bar(range(top_n), probabilities[top_indices], color='steelblue', alpha=0.7)
        axes[0, 1].set_xlabel('Rank (by frequency)', fontsize=12)
        axes[0, 1].set_ylabel('Probability', fontsize=12)
        axes[0, 1].set_title(f'Top {top_n} most common p values', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. 累积分布函数 (CDF)
        sorted_p = np.sort(all_p_values)
        cdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
        axes[1, 0].plot(sorted_p, cdf, linewidth=2)
        axes[1, 0].set_xlabel('p value', fontsize=12)
        axes[1, 0].set_ylabel('Cumulative Probability', fontsize=12)
        axes[1, 0].set_title('Cumulative Distribution Function (CDF)', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. p值的对数直方图
        axes[1, 1].hist(np.log10(all_p_values + 1e-100), bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('log10(p value)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Log-scale Histogram', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_filename = f'./data/{type}_p_distribution_n{n}_layer{layer}_samples{samples}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {plot_filename}")
        
        plt.show()
        
        return all_p_values, unique_values, counts, probabilities

def experiment_unitary_3rdFP(n_start, n_end, layer, samples, type='LC'):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if rank == 0:
        fp = []
    
    for n in range(n_start, n_end+1):
        finite_field = FiniteField(2, n)
        np.random.seed(rank + n)
        # 把 samples 尽量平均分到各个 rank（处理不能整除的情况）
        local_samples = samples // size + (1 if rank < (samples % size) else 0)

        local_sum = 0.0
        if rank == 0:
            for _ in range(local_samples):
                if layer == 0:
                    local_sum += traceU(finite_field, n, layer, type=type, with_identity=False)**3
                else:
                    local_sum += traceU(finite_field, n, layer, type=type)**3
        else:
            for _ in range(local_samples):
                if layer == 0:
                    local_sum += traceU(finite_field, n, layer, type=type, with_identity=False)**3
                else:
                    local_sum += traceU(finite_field, n, layer, type=type)**3
        
        all_result = comm.gather(local_sum, root=0)
        if rank == 0:
            if layer == 0:
                all_result = np.array(all_result)
                if n <= 6:
                    sum_result = np.sum(all_result) / samples * pow(4, 3*n) * (1 - 1 / (pow(2, 5*n) - pow(2, 3*n)))
                else:
                    sum_result = np.sum(all_result) / samples * pow(4, 3*n)
                sum_result += pow(2, 3*n) / (pow(2, 2*n) - 1)
            else:
                all_result = np.array(all_result)
                sum_result = np.sum(all_result) / samples * pow(4, 3*n)
            haar_fp = 6
            result = sum_result / haar_fp - 1
            fp.append(result)
            print(f"n={n} done, result={result}")
    
    if rank == 0:
        # np.save(f'./data/{type}_unitary_3rdFP_{n_start}To{n_end}_layer{layer}_samples{samples}.npy', np.array(fp))
        print(fp)


if __name__ == "__main__":
    # experiment_LC(20, 20, 1, 1000)
    # experiment_H(3, 3, 1, 5)
    # experiment_unitary_3rdFP(3, 3, 0, 1, type='LC')
    
    # 统计p值分布
    collect_p_distribution(n=5, layer=1, samples=5000, type='LC')
