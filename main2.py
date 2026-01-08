from hmac import new
# from mpi4py import MPI
import numpy as np
# from randomSL_sage import random_SL_transformation
from randomSL import *
# from magic_state import *
from stabilizer_state import *
import time
import utils
from tqdm import tqdm
try:
    from mpi4py import MPI
except Exception:
    MPI = None

# prepare initial state
def GHZ(d,n):
    # in tableau basis
    stabilizers = []
    sx = np.concatenate((np.zeros(n,dtype=int), np.ones(n,dtype=int)))
    stabilizers.append(sx)
    if d<=n:
        for j in range(n-1):
            sz = np.zeros(2*n,dtype=int)
            sz[j] = d-1
            sz[n-1] = 1
            stabilizers.append(sz)
    else:
        for j in range(n-1):
            sz = np.concatenate((np.ones(n,dtype=int), np.zeros(n,dtype=int)))
            sz[j] = d-n+1
            stabilizers.append(sz)
    stabVecs = np.array(stabilizers,dtype=int)
    phaseVec = np.zeros(n,dtype=int)
    return stabVecs, phaseVec

def zero(d,n):
    stabVecs = np.concatenate([np.identity(n,dtype=int),np.zeros((n,n),dtype=int)], axis=1)
    phaseVec = np.zeros(n,dtype=int)
    return stabVecs, phaseVec


def restricted_Clifford_update(stabVecs,finite_field,n,m):
    stabZ, stabX = stabVecs[:,:n], stabVecs[:,m:m+n]
    ZvList = [list(stabZ[i]) for i in range(len(stabZ))]
    XvList = [list(stabX[i]) for i in range(len(stabX))]
    new_XvList, new_ZvList = random_SL_transformation(finite_field,XvList,ZvList)
    new_stabX, new_stabZ = [], []
    for i in range(len(new_XvList)):
        new_stabX.append(np.array(new_XvList[i],dtype=int))
        new_stabZ.append(np.array(new_ZvList[i],dtype=int))
    new_stabX, new_stabZ = np.array(new_stabX,dtype=int), np.array(new_stabZ,dtype=int)
    k = stabVecs.shape[0]
    new_stabVecs = np.zeros((k,2*n), dtype=int)
    new_stabVecs[:,:n], new_stabVecs[:,m:m+n] = new_stabZ, new_stabX
    new_stabVecs[:,n:m], new_stabVecs[:,m+n:2*m] = stabVecs[:,n:m], stabVecs[:,m+n:2*m]
    return new_stabVecs

def restricted_Clifford_update_all(stabVecs,finite_field,n):
    stabZ, stabX = stabVecs[:,:n], stabVecs[:,n:2*n]
    ZvList = [list(stabZ[i]) for i in range(len(stabZ))]
    XvList = [list(stabX[i]) for i in range(len(stabX))]
    new_Xvlists, new_Zvlists = SL_transformation_all(finite_field,XvList,ZvList)
    new_stabVecs = []
    for i in range(len(new_Xvlists)):
        new_stabX = np.array(new_Xvlists[i], dtype=int)
        new_stabZ = np.array(new_Zvlists[i], dtype=int)
        k = stabVecs.shape[0]
        new_stabVec = np.zeros((k,2*n), dtype=int)
        new_stabVec[:,:n], new_stabVec[:,n:2*n] = new_stabZ, new_stabX
        new_stabVecs.append(new_stabVec)
    return new_stabVecs

# 域向量化版本：保留原函数，新增更高效的实现以便对比与逐步迁移
def restricted_Clifford_update_all_vec(stabVecs, finite_field, n):
    F = finite_field.field
    k = stabVecs.shape[0]
    # 将整数系数转换为域数组（形状 k x n）
    stabZ = F(stabVecs[:, :n])
    stabX = F(stabVecs[:, n:2*n])

    # 计算每行的域元素 x = sum_j poly_basis[j] * X[j], z = sum_j dual_basis[j] * Z[j]
    # 利用广播与逐列乘加实现向量化
    x_field = F.Zeros(k)
    z_field = F.Zeros(k)
    for j in range(n):
        x_field += finite_field.polynomial_basis[j] * stabX[:, j]
        z_field += finite_field.dual_basis[j] * stabZ[:, j]

    # 获取（并缓存）所有 SL(2, F) 元素
    sl_elements = finite_field.get_sl2_elements()

    new_stabVecs = []
    for mat in sl_elements:
        a = mat[0, 0]
        b = mat[0, 1]
        c = mat[1, 0]
        d = mat[1, 1]
        # 对所有行一次性计算新域元素
        new_x = a * x_field + b * z_field
        new_z = c * x_field + d * z_field

        # 将域元素转换回整数系数（多项式/对偶基表示）
        new_stabX = np.empty((k, n), dtype=int)
        new_stabZ = np.empty((k, n), dtype=int)
        for r in range(k):
            new_stabX[r] = np.array(finite_field.express_in_polynomial_basis(new_x[r]), dtype=int)
            new_stabZ[r] = np.array(finite_field.express_in_dual_basis(new_z[r]), dtype=int)

        new_stabVec = np.zeros((k, 2*n), dtype=int)
        new_stabVec[:, :n] = new_stabZ
        new_stabVec[:, n:2*n] = new_stabX
        new_stabVecs.append(new_stabVec)

    return new_stabVecs
        

# Stabilizer state
def experiment_qudit_stab(d,n,UNum,repNum):
    finite_field = FiniteField(d,n)
    stabVecs, phaseVec = GHZ(d,n)
    divideList = compute_divideList(d)
    fidelities = []
    for _ in range(repNum):
        mean = 0
        for _ in range(UNum):
            # initial state after a restricted Clifford transformation
            new_stabVecs = restricted_Clifford_update(stabVecs,finite_field,n,n)
            rho1 = StabStated(d,n,new_stabVecs,phaseVec.copy(),divideList)
            a = np.random.randint(d, size=2*n, dtype=int)
            rho1.phase_update(a)
            p = rho1.zbasis_measurement()
            f = (pow(d,n)+1)*p-1
            mean+=f
        fidelities.append(mean/UNum)
    return fidelities

def experiment_qubit_stab(n,UNum,repNum):
    stabVecs, phaseVec = GHZ(2,n)
    fidelities = []
    for _ in range(repNum):
        mean = 0
        for _ in range(UNum):
            new_stabVecs = restricted_Clifford_update(stabVecs,finite_field,n,n)
            rho1 = StabState2(n,new_stabVecs,phaseVec.copy())
            a = np.random.randint(2, size=2*n, dtype=int)
            rho1.phase_update(a)
            p = rho1.zbasis_measurement()
            f = (pow(2,n)+1)*p-1
            mean+=f
        fidelities.append(mean/UNum)
    return fidelities

def experiment_qubit_stab_multilayer(rho0,finite_field,m,UNum,repNum):
    n = rho0.n
    fidelities = []
    for _ in tqdm(range(repNum)):
        mean = 0
        for _ in range(UNum):
            new_stabVecs = restricted_Clifford_update(rho0.stabVecs.copy(),finite_field,n,n)
            rho1 = StabState2(n,new_stabVecs,rho0.phaseVec.copy())
            a = np.random.randint(2, size=2*n, dtype=int)
            rho1.phase_update(a)
            for _ in range(m):
                flag = np.random.randint(2,size=n)
                while np.all(flag==0):
                    flag = np.random.randint(2,size=n)
                for k in range(n):
                    if flag[k]: rho1.Hadamard(k)
                new_stabVecs2 = restricted_Clifford_update(rho1.stabVecs,finite_field,n,n)
                rho1.stabVecs = new_stabVecs2
                a = np.random.randint(2, size=2*n, dtype=int)
                rho1.phase_update(a)
            p = rho1.zbasis_measurement()
            # if p == 1:
            #     print(flag)
            #     print(new_stabVecs)
            #     print(new_stabVecs2)
            #     print(rho1.stabVecs, rho1.phaseVec)
            #     exit()
            f = (pow(2,n)+1)*p-1
            mean+=f
        fidelities.append(mean/UNum)
    return np.array(fidelities)

def Hadamard(stabVec, n, it):
    stabVec[:,it], stabVec[:,n+it] = stabVec[:,n+it].copy(), stabVec[:,it].copy()
    return stabVec

def Hadamards(stabVec, n, mask):
    """
    Vectorized Hadamard on multiple qubits.
    - stabVec: (k, 2n) tableau rows
    - n: number of qubits
    - mask: iterable/array of length n; True at positions to apply H
    Returns a new array with columns swapped for all marked qubits.
    """
    m = np.asarray(mask, dtype=bool)
    if m.size != n:
        raise ValueError("mask length must equal n")
    if not m.any():
        return stabVec
    cols = np.arange(2*n)
    cols_swapped = cols.copy()
    idx = np.nonzero(m)[0]
    cols_swapped[idx] = cols[idx + n]
    cols_swapped[idx + n] = cols[idx]
    return stabVec[:, cols_swapped]


def qubit_stab_hadamard_all(n):
    stabVecs, phaseVec = zero(2,n)
    finite_field = FiniteField(2,n)
    # rho0 = StabState2(n, stabVecs, phaseVec.copy())
    fidelities = []
    new_stabVecs_list = restricted_Clifford_update_all_vec(stabVecs,finite_field,n)
    for idx, new_stabVecs in enumerate(new_stabVecs_list):
        # rho1 = StabState2(n,new_stabVecs,phaseVec.copy())
        # a = np.random.randint(2, size=2*n, dtype=int)
        # rho1.phase_update(a)
        for k in range(1 << n):
            new_stabVecsh = new_stabVecs.copy()
            flag = [(k >> i) & 1 for i in range(n)]
            new_stabVecsh = Hadamards(new_stabVecsh, n, flag)
            new_stabVecs_list2 = restricted_Clifford_update_all_vec(new_stabVecsh,finite_field,n)
            for new_stabVecs2 in new_stabVecs_list2:
                rho2 = StabState2(n,new_stabVecs2, phaseVec.copy())
                # a = np.random.randint(2, size=2*n, dtype=int)
                # rho2.phase_update(a)
                p = rho2.zbasis_measurement()
                f = (pow(2,n)+1)*p-1
                fidelities.append(f)
        print(f'{idx} / {len(new_stabVecs_list)} done.')
    return np.array(fidelities)

    

def experiment_qubit_stab_XYZ(rho0,finite_field,UNum,repNum):
    n = rho0.n
    fidelities = []
    for _ in tqdm(range(repNum)):
        mean = 0
        for _ in range(UNum):
            new_stabVecs = restricted_Clifford_update(rho0.stabVecs.copy(),finite_field,n,n)
            rho1 = StabState2(n,new_stabVecs,rho0.phaseVec.copy())
            a = np.random.randint(2, size=2*n, dtype=int)
            rho1.phase_update(a)
            flag = np.random.randint(6,size=n)
            for k in range(n):
                if flag[k]==1:
                    rho1.Hadamard(k)
                elif flag[k]==2:
                    rho1.Phase(k)
                elif flag[k]==3:
                    rho1.Phase(k)
                    rho1.Hadamard(k)
                elif flag[k]==4:
                    rho1.Hadamard(k)
                    rho1.Phase(k)
                elif flag[k]==5:
                    rho1.Hadamard(k)
                    rho1.Phase(k)
                    rho1.Hadamard(k)     
            new_stabVecs = restricted_Clifford_update(rho1.stabVecs,finite_field,n,n)
            rho1.stabVecs = new_stabVecs
            a = np.random.randint(2, size=2*n, dtype=int)
            rho1.phase_update(a)
            p = rho1.zbasis_measurement()
            f = (pow(2,n)+1)*p-1
            mean+=f
        fidelities.append(mean/UNum)
    return np.array(fidelities)

# n = 50
# stabVecs, phaseVec = zero(2,n)
# finite_field = FiniteField(2,n)
# new_stabVecs = restricted_Clifford_update(stabVecs,finite_field,n,n)
# rho0 = StabState2(n,new_stabVecs,phaseVec.copy())
# a = np.random.randint(2, size=2*n, dtype=int)
# rho0.phase_update(a)
# fide = experiment_qubit_stab_XYZ(rho0,finite_field,1,5)
# print(fide)

def experiment_qubit_stab_XZ(rho0,h,UNum,repNum):
    n = rho0.n
    fidelities = []
    finite_field = FiniteField(2,n)
    for _ in range(repNum):
        mean = 0
        for _ in range(UNum):
            new_stabVecs = restricted_Clifford_update(rho0.stabVecs.copy(),finite_field,n,n)
            rho1 = StabState2(n,new_stabVecs,rho0.phaseVec.copy())
            a = np.random.randint(2, size=2*n, dtype=int)
            rho1.phase_update(a)
            positions = np.random.choice(n,h,replace=False)
            for k in positions:
                rho1.Hadamard(k)
            new_stabVecs = restricted_Clifford_update(rho1.stabVecs,finite_field,n,n)
            rho1.stabVecs = new_stabVecs
            a = np.random.randint(2, size=2*n, dtype=int)
            rho1.phase_update(a)
            p = rho1.zbasis_measurement()
            f = (pow(2,n)+1)*p-1
            mean+=f
        fidelities.append(mean/UNum)
    return np.array(fidelities)

# Magic state
def experiment_qudit_magic(d,n,UNum,repNum,Torders1,Torders2):
    fidelities = []
    finite_field = FiniteField(d,n)
    stabVecs, phaseVec = GHZ(d,n)
    WT_dict = compute_WT(d)
    divideList = compute_divideList(d)
    rho0 = StabStated(d,n,stabVecs,phaseVec,divideList)
    t1 = len(Torders1)
    t2 = len(Torders2)
    rho0.add_qudit(t1)
    for i in range(t1):
        rho0.CX(i,n+i) 
    for _ in range(repNum):
        mean = 0
        for run in range(UNum):  
            rho = rho0.copy()
            new_stabVecs = restricted_Clifford_update(rho.stabVecs,finite_field,n,n+t1)
            rho.stabVecs = new_stabVecs
            an = np.random.randint(d, size=2*n, dtype=int)
            a = np.zeros(2*(n+t1),dtype=int)
            a[0:n],a[n+t1:2*n+t1] = an[0:n],an[n:2*n]
            rho.phase_update(a)
            rho.add_qudit(t2)
            for i in range(t2):
                rho.CX(i,n+t1+i)
                rho.Hadamard(i)
            rhom = MagicState(d,n,rho.stabVecs,rho.phaseVec,np.array(Torders1+Torders2))
            rhom.preprocess()
            res,pxs = rhom.sample_outcome(1,WT_dict,divideList)
            f = (pow(d,n)+1)*pxs[0]-1
            mean+=f
        fidelities.append(mean/UNum)
    return np.array(fidelities)

def qudit_magic_main(d,n):
    UNum = 1
    repNum = 50000
    Torders1 = []
    Torders2 = [0]
    fidelities = experiment_qudit_magic(d,n,UNum,repNum,Torders1,Torders2)
    np.save(f'0319/d={d},n={n},t1={len(Torders1)},t2={len(Torders2)},N={repNum},GHZ.npy',fidelities)


def experiment_qubit_magic(n,Unum,t1,t2):
    stabVecs, phaseVec = GHZ(2,n)
    WT_dict = compute_WT(2)
    finite_field = FiniteField(2,n)
    rho0 = StabState2(n,stabVecs,phaseVec)
    rho0.add_qubit(t1)
    for i in range(t1):
        rho0.CNOT(i,n+i)
    mean = 0
    for _ in range(Unum): 
        rho = rho0.copy()
        new_stabVecs = restricted_Clifford_update(rho.stabVecs,finite_field,n,n+t1)
        rho.stabVecs = new_stabVecs
        an = np.random.randint(2, size=2*n, dtype=int)
        a = np.zeros(2*(n+t1),dtype=int)
        a[0:n],a[n+t1:2*n+t1] = an[0:n],an[n:2*n]
        rho.phase_update(a)
        rho.add_qubit(t2)
        for i in range(t2):
            rho.CNOT(i,n+t1+i)
            rho.Hadamard(i)
        rhom = QubitMagic(n,t1+t2,rho.stabVecs,rho.phaseVec)
        rhom.step1()
        rhom.step2()
        rhom.step3()
        res,pxs = rhom.sample_outcome(1,WT_dict)
        f = (pow(2,n)+1)*pxs[0]-1
        mean += f
    return mean/Unum

def main_XYZ(d,n,idx):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    UNum, samNum, repNum = 1, 1, 10000
    stabVecs, phaseVec = zero(2,n)
    finite_field = FiniteField(d,n)
    rho0 = StabState2(n, stabVecs, phaseVec.copy())
    np.random.seed(rank + n)
    # 每个进程处理的实验次数
    local_repNum = repNum // size
    local_res = experiment_qubit_stab_XYZ(rho0, finite_field, UNum, local_repNum)
    
    # 收集结果
    combine_result = comm.gather(local_res, root=0)

    if rank == 0:
        combine_result = np.array(combine_result)
        output = combine_result.flatten()
        np.save(f'./data/XYZ/d={d},n={n},XYZ.npy', output)


def main_multilayer(d,n,idx):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    UNum, samNum, repNum = 1, 1, 5000
    stabVecs, phaseVec = zero(2,n)
    finite_field = FiniteField(d,n)
    rho0 = StabState2(n, stabVecs, phaseVec.copy())
    np.random.seed(rank + n)
    # 每个进程处理的实验次数
    local_repNum = repNum // size
    for m in [1]:
        local_res = experiment_qubit_stab_multilayer(rho0, finite_field, m, UNum, local_repNum)
        combine_result = comm.gather(local_res, root=0)

        if rank == 0:
            combine_result = np.array(combine_result)
            output = combine_result.flatten()
            np.save(f'./data/Hadamard/d={d},n={n},m={m}_{idx}.npy',output)


def main_XZ(d,n,idx):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    UNum,samNum,repNum = 1,1,2000
    stabVecs, phaseVec = zero(2,n)
    finite_field = FiniteField(d,n)
    np.random.seed(rank + n)
    new_stabVecs = restricted_Clifford_update(stabVecs,finite_field,n,n)
    rho0 = StabState2(n,new_stabVecs,phaseVec.copy())
    a = np.random.randint(2, size=2*n, dtype=int)
    rho0.phase_update(a)
    for h in range(n+1):
        res = experiment_qubit_stab_XZ(rho0,h,UNum,repNum)
        combine_result = comm.gather(res, root = 0)

        if rank==0:
            combine_result = np.array(combine_result)
            output = combine_result.flatten()
            np.save(f'1226/singlelayer/d={d},n={n},h={h},XYZ,rand{idx}.npy',output)


if __name__=='__main__':
    # for n in range(2, 3):
    #     main_XYZ(2,n,1)
    UNum, samNum, repNum = 1, 1, 50000
    n=4
    d=2
    stabVecs, phaseVec = zero(2,n)
    finite_field = FiniteField(d,n)
    rho0 = StabState2(n, stabVecs, phaseVec.copy())
    # np.random.seed(rank + n)
    # 每个进程处理的实验次数
    result = experiment_qubit_stab_XYZ(rho0, finite_field, UNum, repNum)
    np.save(f'./data/XYZ/d={d},n={n},sample={repNum},XYZ.npy', result)
    # for n in range(3, 4):
        # main_multilayer(2,n,2)
    # for idx in range(20):
    #     print(f'Run {idx} now!')
    #     main_multilayer(2,3,idx)
    # time_start = time.time()
    # all_result = qubit_stab_hadamard_all(3)
    # print(f'mean = {np.mean(all_result)}, var = {np.var(all_result)}')
    # np.save(f'./data/Hadamard/d={2},n={3},all.npy', all_result)
    # time_end = time.time()
    # print(f'Time used: {time_end - time_start} seconds.')

    # all_result = qubit_stab_hadamard_all(3)
    # print(f'mean = {np.mean(all_result)}, var = {np.var(all_result)}')
    # np.save(f'./data/Hadamard/d={2},n={3},all.npy', all_result)

