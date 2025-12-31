from mpi4py import MPI
import numpy as np
# from randomSL_sage import random_SL_transformation
from randomSL import *
from magic_state import *
from qubit_magic import *
import time

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
    new_stabVecs = np.zeros((m,2*m), dtype=int)
    new_stabVecs[:,:n], new_stabVecs[:,m:m+n] = new_stabZ, new_stabX
    new_stabVecs[:,n:m], new_stabVecs[:,m+n:2*m] = stabVecs[:,n:m], stabVecs[:,m+n:2*m]
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
    for _ in range(repNum):
        mean = 0
        for _ in range(UNum):
            new_stabVecs = restricted_Clifford_update(rho0.stabVecs.copy(),finite_field,n,n)
            rho1 = StabState2(n,new_stabVecs,rho0.phaseVec.copy())
            a = np.random.randint(2, size=2*n, dtype=int)
            rho1.phase_update(a)
            for _ in range(m):
                flag = np.random.randint(2,size=n)
                for k in range(n):
                    if flag[k]: rho1.Hadamard(k)
                new_stabVecs = restricted_Clifford_update(rho1.stabVecs,finite_field,n,n)
                rho1.stabVecs = new_stabVecs
                a = np.random.randint(2, size=2*n, dtype=int)
                rho1.phase_update(a)
            p = rho1.zbasis_measurement()
            f = (pow(2,n)+1)*p-1
            mean+=f
        fidelities.append(mean/UNum)
    return np.array(fidelities)

def experiment_qubit_stab_XYZ(rho0,finite_field,UNum,repNum):
    n = rho0.n
    fidelities = []
    for _ in range(repNum):
        mean = 0
        for _ in range(UNum):
            new_stabVecs = restricted_Clifford_update(rho0.stabVecs.copy(),finite_field,n,n)
            rho1 = StabState2(n,new_stabVecs,rho0.phaseVec.copy())
            a = np.random.randint(2, size=2*n, dtype=int)
            rho1.phase_update(a)
            flag = np.random.randint(3,size=n)
            for k in range(n):
                if flag[k]==1:
                    rho1.Hadamard(k)
                elif flag[k]==2:
                    rho1.Phase(k)
                    rho1.Hadamard(k)
            new_stabVecs = restricted_Clifford_update(rho1.stabVecs,finite_field,n,n)
            rho1.stabVecs = new_stabVecs
            a = np.random.randint(2, size=2*n, dtype=int)
            rho1.phase_update(a)
            rho1.check_commutator()
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
    
    UNum,samNum,repNum = 1,1,2000
    stabVecs, phaseVec = zero(2,n)
    finite_field = FiniteField(d,n)
    new_stabVecs = restricted_Clifford_update(stabVecs,finite_field,n,n)
    rho0 = StabState2(n,new_stabVecs,phaseVec.copy())
    a = np.random.randint(2, size=2*n, dtype=int)
    rho0.phase_update(a)
    res = experiment_qubit_stab_XYZ(rho0,finite_field,UNum,repNum)
    combine_result = comm.gather(res, root = 0)

    if rank==0:
        combine_result = np.array(combine_result)
        output = combine_result.flatten()
        np.save(f'1226/XYZ/d={d},n={n},XYZ,rand{idx}.npy',output)



def main_multilayer(d,n,idx):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    UNum,samNum,repNum = 1,1,2000
    stabVecs, phaseVec = zero(2,n)
    finite_field = FiniteField(d,n)
    new_stabVecs = restricted_Clifford_update(stabVecs,finite_field,n,n)
    rho0 = StabState2(n,new_stabVecs,phaseVec.copy())
    a = np.random.randint(2, size=2*n, dtype=int)
    rho0.phase_update(a)
    for m in [1,2]:
        res = experiment_qubit_stab_multilayer(rho0,finite_field,m,UNum,repNum)
        combine_result = comm.gather(res, root = 0)

        if rank==0:
            combine_result = np.array(combine_result)
            output = combine_result.flatten()
            np.save(f'1226/Hadamard/d={d},n={n},m={m},rand{idx}.npy',output)


def main_XZ(d,n,idx):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    UNum,samNum,repNum = 1,1,2000
    stabVecs, phaseVec = zero(2,n)
    finite_field = FiniteField(d,n)
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
    for idx in range(1):
        main_XYZ(2,5,idx)
    # for idx in range(1):
    #     main_multilayer(2,5,idx)
    

