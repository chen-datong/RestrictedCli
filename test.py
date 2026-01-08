import numpy as np
from randomSL import random_SL_transformation
from magic_state import *
from qubit_magic import *
import randomSL

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

def restricted_Clifford_update(stabVecs,d,n,m):
    stabZ, stabX = stabVecs[:,:n], stabVecs[:,m:m+n]
    print(f'stabZ = {stabZ}, stabX = {stabX}')
    ZvList = [list(stabZ[i]) for i in range(len(stabZ))]
    XvList = [list(stabX[i]) for i in range(len(stabX))]
    new_XvList, new_ZvList = random_SL_transformation(d,n,XvList,ZvList)
    new_stabX, new_stabZ = [], []
    for i in range(len(new_XvList)):
        new_stabX.append(np.array(new_XvList[i],dtype=int))
        new_stabZ.append(np.array(new_ZvList[i],dtype=int))
    new_stabX, new_stabZ = np.array(new_stabX,dtype=int), np.array(new_stabZ,dtype=int)
    print(f'new_stabZ = {new_stabZ}, new_stabX = {new_stabX}')
    new_stabVecs = np.zeros((m,2*m), dtype=int)
    new_stabVecs[:,:n], new_stabVecs[:,m:m+n] = new_stabZ, new_stabX
    new_stabVecs[:,n:m], new_stabVecs[:,m+n:2*m] = stabVecs[:,n:m], stabVecs[:,m+n:2*m]
    return new_stabVecs

# d = 3
# n = 5
# stabVecs, phaseVec = GHZ(d,n)
# divideList = compute_divideList(d)
# new_stabVecs = restricted_Clifford_update(stabVecs,d,n,n)
# rho1 = StabStated(d,n,new_stabVecs,phaseVec,divideList)
# rho1.check_commutator()
# d = 2
# n = 4
# finite_field = randomSL.FiniteField(d, n)
# randomSL.check_sl_embedding_is_symplectic(finite_field, trials=10)