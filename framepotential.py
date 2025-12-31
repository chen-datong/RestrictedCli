from mpi4py import MPI
import numpy as np
from randomSL import *
from magic_state import *
from main2 import restricted_Clifford_update

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
    rho = StabState2(n,new_stabVecs,phaseVec)
    a = np.random.randint(2, size=2*n, dtype=int)
    rho.phase_update(a)
    for _ in range(layer):
        flag = np.random.randint(3,size=n)
        for k in range(n):
            if flag[k]==1:
                rho.Hadamard(k)
            elif flag[k]==2:
                rho.Phase(k)
                rho.Hadamard(k)
        new_stabVecs = restricted_Clifford_update(rho.stabVecs,finite_field,n,n)
        rho.stabVecs = new_stabVecs
        a = np.random.randint(2, size=2*n, dtype=int)
        rho.phase_update(a)
    p = rho.inner_zero()
    return pow(p, 3)

def LC_3rdFP(n, layer, samples):
    finite_field = FiniteField(2, n)
    sum = 0
    for _ in range(samples):
        sum += single_LC_3rdFP(n, layer, finite_field)
    sum /= samples
    haar_fp = Haar_3rdFP(n)
    return sum / haar_fp - 1

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
    for _ in range(layer):
        flag = np.random.randint(2,size=n)
        for k in range(n):
            if flag[k]: rho.Hadamard(k)
        new_stabVecs = restricted_Clifford_update(rho.stabVecs,finite_field,n,n)
        rho.stabVecs = new_stabVecs
        a = np.random.randint(2, size=2*n, dtype=int)
        rho.phase_update(a)
    p = rho.inner_zero()
    return pow(p, 3)

def H_3rdFP(n, layer, samples):
    finite_field = FiniteField(2, n)
    sum = 0
    for _ in range(samples):
        sum += single_LC_3rdFP(n, layer, finite_field)
    sum /= samples
    haar_fp = Haar_3rdFP(n)
    return sum / haar_fp - 1

def experiment_LC(n_start, n_end, layer, samples):
    fp = []
    for n in range(n_start, n_end+1):
        fp.append(LC_3rdFP(n, layer, samples))
    print(fp)

if __name__ == "__main__":
    experiment_LC(3, 10, 1, 1000)
