import numpy as np
from utils import *

class StabStated:

    def __init__(self,d,n,stabVecs,phaseVec,divideList):
        self.d = d
        self.n = n
        self.divideList = divideList
        self.stabVecs = stabVecs 
        self.phaseVec = phaseVec

    def copy(self):
        newState = StabStated(self.d,self.n,self.stabVecs.copy(),self.phaseVec.copy(),self.divideList.copy())
        return newState

    def check_commutator(self):
        print(commutator_t(self.stabVecs,np.transpose(self.stabVecs),self.d,self.n))

    def phase_update(self,a):
        self.phaseVec = (self.phaseVec + commutator_t(a,np.transpose(self.stabVecs),self.d,self.n))%self.d

    def zbasis_measurement(self):
        L0 = np.concatenate([np.identity(self.n,dtype=int),np.zeros((self.n,self.n),dtype=int)],axis=1)
        _, D = Zassenhaus(L0,self.stabVecs,self.d)
        rownum = np.shape(D)[0]
        p = pow(self.d,rownum-self.n)
        return p

    def add_qudit(self,k):
        # add k qudits in |0>
        self.stabVecs = np.concatenate([self.stabVecs[:,:self.n],np.zeros((self.n,k),dtype=int), self.stabVecs[:,self.n:],np.zeros((self.n,k),dtype=int)],axis=1)
        newstab = np.concatenate([np.zeros((k,self.n),dtype=int),np.identity(k,dtype=int),np.zeros((k,self.n+k),dtype=int)],axis=1)
        self.stabVecs = np.concatenate([self.stabVecs,newstab])
        self.phaseVec = np.concatenate([self.phaseVec,np.zeros(k,dtype=int)])
        self.n = self.n+k

    def CX(self,ic,it):
        # ic: index of control qudit
        # it: index of target qudit
        self.stabVecs[:,ic],self.stabVecs[:,self.n+it] = \
            self.stabVecs[:,ic].copy()-self.stabVecs[:,it].copy(),self.stabVecs[:,self.n+it].copy()+self.stabVecs[:,self.n+ic].copy()
        self.stabVecs = (self.stabVecs)%self.d

    def Hadamard(self,it):
        self.stabVecs[:,it],self.stabVecs[:,self.n+it] =\
             self.stabVecs[:,self.n+it].copy(),-self.stabVecs[:,it].copy()
        self.stabVecs = (self.stabVecs)%self.d


class StabState2:

    def __init__(self,n,stabVecs,phaseVec):
        self.n = n
        self.stabVecs = stabVecs # [Z|X] convention
        self.phaseVec = phaseVec

    def copy(self):
        newState = StabState2(self.n,self.stabVecs.copy(),self.phaseVec.copy())
        return newState

    def check_commutator(self):
        print(commutator_t(self.stabVecs,np.transpose(self.stabVecs),2,self.n))

    def phase_update(self,a):
        for i in range(self.n):
            for j in range(self.n):
                P = np.array([self.stabVecs[i,j], self.stabVecs[i,self.n+j]], dtype=int)
                aj = np.array([a[j], a[self.n+j]], dtype=int)
                if P.any()!=0 and (P-aj).any()!=0:
                    self.phaseVec[i] += 1
        self.phaseVec = self.phaseVec%2

    def zbasis_measurement(self):
        # M = self.stabVecs[:,:self.n]
        M = self.stabVecs[:,self.n: 2*self.n].copy()
        i,j = 0,0
        while i<self.n and j<self.n:
            if M[i,j]==0:
                for k in range(i+1,self.n):
                    if M[k,j]!=0:
                        M[i], M[k] = M[k].copy(), M[i].copy()
                        break
                else:
                    j += 1
                    continue
            # forward elimination
            for k in range(i+1,self.n):
                if M[k,j]==1:
                    M[k] = (M[k]+M[i])%2
            i += 1
            j += 1
        rank = i
        p = 1/pow(2,rank)
        return p

    def add_qubit(self,k):
        # add k qudits in |0>
        self.stabVecs = np.concatenate([self.stabVecs[:,:self.n],np.zeros((self.n,k),dtype=int),\
                        self.stabVecs[:,self.n:],np.zeros((self.n,k),dtype=int)],axis=1)
        newstab = np.concatenate([np.zeros((k,self.n),dtype=int),np.identity(k,dtype=int),np.zeros((k,self.n+k),dtype=int)],axis=1)
        self.stabVecs = np.concatenate([self.stabVecs,newstab])
        self.phaseVec = np.concatenate([self.phaseVec,np.zeros(k,dtype=int)])
        self.n = self.n+k
    
    def CNOT(self,ic,it):
        self.phaseVec = (self.phaseVec+self.stabVecs[:,self.n+ic]*self.stabVecs[:,it]*\
            (self.stabVecs[:,self.n+it]+self.stabVecs[:,ic]+np.ones(self.n,dtype=int))%2)%2
        self.stabVecs[:,ic], self.stabVecs[:,self.n+it] = (self.stabVecs[:,ic].copy()+self.stabVecs[:,it].copy())%2,\
            (self.stabVecs[:,self.n+ic].copy()+self.stabVecs[:,self.n+it].copy())%2
            
    def Hadamard(self,it):
        self.phaseVec = (self.phaseVec+self.stabVecs[:,self.n+it]*self.stabVecs[:,it])%2
        self.stabVecs[:,it], self.stabVecs[:,self.n+it] = self.stabVecs[:,self.n+it].copy(), self.stabVecs[:,it].copy()

    def Phase(self, it):
        # for r in range(self.n):
        #     if self.stabVecs[r,self.n+it]==0:   pass 
        #     else:
        #         if self.stabVecs[r,self.n+it]==0:
        #             self.stabVecs[r,self.n+it] = 1
        #         else:
        #             self.stabVecs[r,self.n+it] = 0
        #             self.phaseVec[r] = (self.phaseVec[r]+1)%2
        self.phaseVec = (self.phaseVec+self.stabVecs[:,self.n+it]*self.stabVecs[:,it])%2
        self.stabVecs[:, it] = (self.stabVecs[:, it] + self.stabVecs[:, self.n + it]) % 2

    def inner_zero(self):
        M = self.stabVecs[:,self.n: 2*self.n].copy()
        # M = self.stabVecs[:,:self.n].copy()
        i,j = 0,0
        while i<self.n and j<self.n:
            if M[i,j]==0:
                for k in range(i+1,self.n):
                    if M[k,j]!=0:
                        M[i], M[k] = M[k].copy(), M[i].copy()
                        self.stabVecs[i], self.stabVecs[k] = self.stabVecs[k].copy(), self.stabVecs[i].copy()
                        self.phaseVec[i], self.phaseVec[k] = self.phaseVec[k].copy(), self.phaseVec[i].copy()
                        break
                else:
                    j += 1
                    continue
            # forward elimination
            for k in range(i+1,self.n):
                if M[k,j]==1:
                    g = rowsum(self.stabVecs[k], self.stabVecs[i], self.n)
                    M[k] = (M[k]+M[i])%2
                    self.stabVecs[k] = (self.stabVecs[k]+self.stabVecs[i])%2
                    temp = 2 * self.phaseVec[k] + 2 * self.phaseVec[i] + g
                    if temp % 4 == 0:
                        self.phaseVec[k] = 0
                    elif temp % 4 == 2:
                        self.phaseVec[k] = 1
                    else: raise Exception('rowsum = 1 or 3!')
            i += 1
            j += 1
        rank = i
        for k in range(i, self.n):
            if self.phaseVec[k] != 0:
                return 0
        return 1 / pow(2, rank)
    

