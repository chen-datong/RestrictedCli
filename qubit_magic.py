import numpy as np
from stabilizer_state import *

class QubitMagic:

    def __init__(self,n,t,Stab,phaseVec):
        self.n = n
        self.t = t
        self.Stab = Stab 
        self.phaseVec = phaseVec
        self.lastSite = []
        self.emptyRows = []

    def check_commutator(self):
        print(commutator_t(self.Stab,np.transpose(self.Stab),self.n+self.t))

    def rowadd(self,h,i):
        # add generator i to generator h
        g = rowsum(self.Stab[h],self.Stab[i],self.n+self.t)
        self.Stab[h] = (self.Stab[h]+self.Stab[i])%2 
        temp = 2*self.phaseVec[h]+2*self.phaseVec[i]+g
        if temp%4==0:   self.phaseVec[h] = 0
        elif temp%4==2: self.phaseVec[h] = 1
        else: raise Exception('rowsum = 1 or 3!')

    def step1(self):
        # step 1: Eliminate all X in register a,b
        x = self.n+self.t
        i,j = 0,0
        while i<x and j<self.n:
            if self.Stab[i,x+j]==0:
                for k in range(i+1,x):
                    if self.Stab[k,x+j]!=0:
                        self.Stab[i], self.Stab[k] = self.Stab[k].copy(), self.Stab[i].copy()
                        self.phaseVec[i], self.phaseVec[k] = self.phaseVec[k], self.phaseVec[i]
                        break
                else:
                    j+=1
                    continue
            # forward elimination
            for k in range(i+1,x):
                if self.Stab[k,x+j]==1:
                    self.rowadd(k,i)
            i += 1
            j += 1
        rank = i
        self.Stab = self.Stab[rank:]
        self.phaseVec = self.phaseVec[rank:]

    def step2(self):
        # step 2: Transform Zab into column-inverse form
        m = len(self.Stab)
        i,j = 0,0
        while i<m and j<self.n:
            if self.Stab[i,self.n-1-j]==0:
                for k in range(i+1,m):
                    if self.Stab[k,self.n-1-j]!=0:
                        self.Stab[i], self.Stab[k] = self.Stab[k].copy(), self.Stab[i].copy()
                        self.phaseVec[i], self.phaseVec[k] = self.phaseVec[k], self.phaseVec[i]
                        break
                else:
                    j += 1
                    continue
            self.lastSite.append(self.n-1-j)
            # forward elimination
            for k in range(i+1,m):
                if self.Stab[k,self.n-1-j]==1:
                    self.rowadd(k,i)
            i += 1
            j += 1
        self.lastSite.extend([-1]*(m-i))
    
    def step3(self):
        # step 3: Transform ZcXc into echelon form
        # only add lower rows to upper rows, row exchange is not allowed
        m = len(self.Stab)
        x = 2*self.n+self.t
        # Zc part
        emptyRows_z = []
        i,j = 0,0
        while i<m and j<self.t:
            if self.Stab[m-1-i,self.n+j]==0:
                for l in range(j+1,self.t):
                    if self.Stab[m-1-i,self.n+l]!=0:
                        self.Stab[:,self.n+j],self.Stab[:,self.n+l] = self.Stab[:,self.n+l].copy(),self.Stab[:,self.n+j].copy()
                        self.Stab[:,x+j],self.Stab[:,x+l] = self.Stab[:,x+l].copy(),self.Stab[:,x+j].copy()
                        break                
                else:
                    emptyRows_z.append(m-1-i)
                    i += 1
                    continue
            # forward elimination
            for k in range(i+1,m):
                if self.Stab[m-1-k,self.n+j]==1:
                    self.rowadd(m-1-k,m-1-i)
            i += 1
            j += 1
        for k in range(i,m):
            emptyRows_z.append(m-1-k)
        # Xc part
        mx = len(emptyRows_z)
        i,j = 0,0
        while i<mx and j<self.t:
            if self.Stab[emptyRows_z[i],x+j]==0:
                for l in range(j+1,self.t):
                    if self.Stab[emptyRows_z[i],x+l]!=0:
                        self.Stab[:,self.n+j],self.Stab[:,self.n+l] = self.Stab[:,self.n+l].copy(),self.Stab[:,self.n+j].copy()
                        self.Stab[:,x+j],self.Stab[:,x+l] = self.Stab[:,x+l].copy(),self.Stab[:,x+j].copy()
                        break                
                else:
                    self.emptyRows.append(emptyRows_z[i])
                    i += 1
                    continue
            # forward elimination
            for k in range(i+1,mx):
                if self.Stab[emptyRows_z[k],x+j]==1:
                    self.rowadd(emptyRows_z[k],emptyRows_z[i])
            i += 1
            j += 1
        self.emptyRows.extend(emptyRows_z[i:mx])
        self.emptyRows.sort()
        self.emptyRows = np.array(self.emptyRows, dtype=int)

    def cut_w(self,w):
        m = len(self.Stab)
        for i in range(m):
            if self.lastSite[i]<w:
                return i
        else:
            return m

    def sample_outcome(self,samNum,WT_dict):
        m1 = len(self.Stab)
        niRows = np.array([i for i in range(m1) if i not in self.emptyRows], dtype=int)
        l = len(niRows)
        if l==0:    GcT_dict = {():1}
        else:
            GcT_dict = {}
            for idx in range(pow(2,l)):
                v = index_to_vector(idx,2,l)
                u = np.zeros(2*(self.n+self.t), dtype=int)
                ru = 0
                for i in range(l):
                    if v[i]==1:
                        temp = 2*self.phaseVec[niRows[i]]+2*ru+rowsum(self.Stab[niRows[i]],u,self.n+self.t)
                        if temp%4==0: ru = 0
                        elif temp%4==2: ru = 1
                        else: raise Exception('rowsum = 1 or 3!')
                        u = (self.Stab[niRows[i]]+u)%2
                val = 1
                for j in range(self.t):
                    val *= WT_dict[(u[self.n+j],u[2*self.n+self.t+j])]
                GcT_dict[tuple(v)] = val*pow(-1,ru)
        
        res = []
        pxs = []
        for _ in range(samNum):
            x = np.array([],dtype=int)
            px = 1
            for w in range(1,self.n+1):
                c2 = self.cut_w(w)
                xi = m1-c2
                iRows_w = [i for i in range(c2,m1) if i in self.emptyRows]
                niRows_w = [i for i in range(c2,m1) if i in niRows]
                m3 = len(niRows_w)
                prob = []
                for y in range(2):
                    xy = np.concatenate([x,np.array([y],dtype=int)])
                    phase0y = np.array([self.phaseVec[i]+np.dot(xy,self.Stab[i,:w]) for i in iRows_w], dtype=int)%2
                    if phase0y.any()!=0:
                        prob.append(0)
                    else:
                        if m3==0:   
                            label = tuple(np.zeros(l,dtype=int))
                            py = GcT_dict[label]
                        else:
                            py = 0
                            for idx in range(pow(2,m3)):
                                v = index_to_vector(idx,2,m3)
                                temp = np.array([self.Stab[i,:w] for i in niRows_w],dtype=int)
                                phi = np.dot(xy,v@temp)
                                label = tuple(np.concatenate([np.zeros(l-m3,dtype=int),v]))
                                py += pow(-1,phi)*GcT_dict[label]
                        py *= pow(2,xi-w-m3)
                        prob.append(np.abs(py.real))
                y = np.random.choice(2, p = np.array(prob)/px)
                px = prob[y]
                x = np.concatenate([x,np.array([y],dtype=int)])
        res.append(x)
        pxs.append(px)
        return res,pxs



                
            










        
