import numpy as np
from stabilizer_state import *
from scipy.linalg import block_diag

class MagicState:

    def __init__(self,d,n,Stab,phaseVec,Torders):
        self.Stab = Stab
        self.Torders = Torders
        self.phaseVec = phaseVec
        self.d = d
        self.n = n
        self.t = len(Torders)
        self.emptyRows = None

    def check_commutator(self):
        C = commutator_t(self.Stab,np.transpose(self.Stab),self.d,self.n+self.t)
        if C.any()==0:  return True
        else:   return False


    def preprocess(self):
        # step 1: Eliminate all X in register a,b
        M = self.Stab[:,self.n+self.t:2*self.n+self.t]
        echM = echelonMatrix(M,self.d)
        echM.echelon()
        Zabc = (echM.Sleft@self.Stab[:,:self.n+self.t])%self.d
        Xc = (echM.Sleft@self.Stab[:,2*self.n+self.t:2*(self.n+self.t)])%self.d
        self.Stab = np.concatenate([Zabc,echM.M,Xc],axis=1)
        self.Stab = self.Stab[echM.rk:]
        self.phaseVec = (echM.Sleft@self.phaseVec)%self.d
        self.phaseVec = self.phaseVec[echM.rk:]

        # step 2: Transform Zab into column-inverse form
        M = self.Stab[:,:self.n]
        M = M[:,-1:-self.n-1:-1] 
        echM = echelonMatrix(M,self.d)
        echM.echelon()
        Zab = echM.M[:,-1:-self.n-1:-1]
        Zc = (echM.Sleft@self.Stab[:,self.n:self.n+self.t])%self.d
        Xabc = (echM.Sleft@self.Stab[:,self.n+self.t:2*(self.n+self.t)])%self.d
        self.Stab = np.concatenate([Zab,Zc,Xabc],axis=1)
        self.phaseVec = (echM.Sleft@self.phaseVec)%self.d
        stabcopy = self.Stab.copy()
        phasecopy = self.phaseVec.copy()

        # step 3: Transform ZcXc into echelon form
        # only add lower rows to upper rows, column exchange is not allowed
        m1,_  = np.shape(self.Stab)
        Zab = self.Stab[:,:self.n]
        Mzc,Mxc = self.Stab[:,self.n:self.n+self.t],self.Stab[:,2*self.n+self.t:2*(self.n+self.t)]
        Mzc,Mxc = Mzc[-1:-m1-1:-1,:],Mxc[-1:-m1-1:-1,:]
        echMzc = echelonMatrix(Mzc,self.d)
        echMzc.echelon(row_exchange=False)
        xcRows = []
        for i in range(m1):
            if i in echMzc.emptyRows or echMzc.M[i].any()==0:
                xcRows.append(i)
        self.Torders = self.Torders@echMzc.Sright
        T1 = np.identity(m1,dtype=int)
        T1 = T1[-1:-m1-1:-1,:] # row inverse
        Zab = (T1@echMzc.Sleft@T1@Zab)%self.d
        self.phaseVec = (T1@echMzc.Sleft@T1@self.phaseVec)%self.d

        Mxc = (echMzc.Sleft@Mxc@echMzc.Sright)%self.d
        Mxc1 = np.concatenate([Mxc[np.newaxis,i] for i in xcRows]) if xcRows else Mxc[0:0]
        echMxc = echelonMatrix(Mxc1,self.d)
        echMxc.echelon(row_exchange=False)
        T2 = np.zeros((m1,m1),dtype=int) # transform [0,1,2,3] to [1,3,0,2] (xcRows=[1,3])
        m2 = len(Mxc1)
        j1,j2=0,0
        for i in range(m1):
            if i in xcRows:
                T2[j1,i] = 1
                j1+=1
            else:
                T2[m2+j2,i] = 1
                j2+=1
        Sxcleft = np.transpose(T2)@block_diag(echMxc.Sleft,np.identity(m1-m2,dtype=int))@T2
        self.Torders = (np.transpose(echMxc.Sright)@self.Torders)%self.d
        Zab = (T1@Sxcleft@T1@Zab)%self.d
        self.phaseVec = (T1@Sxcleft@T1@self.phaseVec)%self.d

        Zc_inv = (Sxcleft@echMzc.M@echMxc.Sright)%self.d
        xcRows1 = [Mxc[np.newaxis,i] for i in range(m1) if i not in xcRows]
        Mxc0 = np.concatenate(xcRows1)@echMxc.Sright if xcRows1 else Mxc[0:0]
        Xc_inv = np.transpose(T2)@np.concatenate([echMxc.M,Mxc0])
        Xab = np.zeros((m1,self.n),dtype=int)
        self.Stab = np.concatenate([Zab,Zc_inv[-1:-m1-1:-1],Xab,Xc_inv[-1:-m1-1:-1]],axis=1)
        # record the empty rows
        temp = np.array([1 if i in echMxc.emptyRows else 0 for i in range(m1)], dtype=int)
        temp = T1@np.transpose(T2)@temp
        self.emptyRows = np.array([i for i in range(m1) if temp[i]==1],dtype=int)


    def cut_w(self,w):
        m1 = len(self.Stab)
        c2 = m1
        # check position w in Zab
        for i in range(m1-1,-1,-1):
            if (self.Stab[i,w:self.n]).any()==0:
                c2 = i
            else:   break
        c3 = m1
        for i in range(c2,m1):
            if (self.Stab[i,self.n:self.n+self.t]).any()!=0 or (self.Stab[i,2*self.n+self.t:2*(self.n+self.t)]).any()!=0:
                c3 = i
                break
        idx0,idx1 = [],[]
        for i in range(c2,m1):
            if i<c3 or i in self.emptyRows: idx0.append(i)
            else:   idx1.append(i)
        idx0,idx1 = np.array(idx0,dtype=int),np.array(idx1,dtype=int)
        subG0 = self.Stab[idx0] if len(idx0)!=0 else self.Stab[0:0]
        subG1 = self.Stab[idx1] if len(idx1)!=0 else self.Stab[0:0]
        Ggamma = np.concatenate([self.Stab[idx1,self.n:self.n+self.t],\
                    self.Stab[idx1,2*self.n+self.t:2*(self.n+self.t)]],axis=1) if len(idx1)!=0 else \
                        np.concatenate([self.Stab[0:0,self.n:self.n+self.t],\
                    self.Stab[0:0,2*self.n+self.t:2*(self.n+self.t)]],axis=1)
        phase0 = self.phaseVec[idx0] if len(idx0)!=0 else self.phaseVec[0:0]
        phase1 = self.phaseVec[idx1] if len(idx1)!=0 else self.phaseVec[0:0]
        return subG0,phase0,Ggamma,subG1,phase1,m1-c2

    
    def compute_prodT(self,Ggamma,WT_dict,divideList):
        GcT_dict = {():1}
        m3 = len(Ggamma)
        for idx in range(pow(self.d,m3)):
            v = index_to_vector(idx,self.d,m3)
            stabvec = (np.transpose(Ggamma)@v)%self.d
            val = 1
            for i in range(self.t):
                key = (stabvec[i],stabvec[self.t+i],self.Torders[i])
                val *= WT_dict[key]
            GcT_dict[tuple(v)] = val
        return GcT_dict


    def sample_outcome(self,samNum,WT_dict,divideList):
        omega = np.exp(2*np.pi*1j/self.d)
        res = []
        pxs = []
        for _ in range(samNum):
            x = np.array([],dtype=int)
            px = 1
            for w in range(1,self.n+1):
                subG0,phase0,Ggamma,subG1,phase1,xi = self.cut_w(w)
                m3 = len(Ggamma)
                GcT_dict = self.compute_prodT(Ggamma,WT_dict,divideList)
                prob = []
                for y in range(self.d):
                    xy = np.concatenate([x,np.array([y],dtype=int)])
                    phase0y = (phase0+subG0[:,:w]@xy)%self.d
                    phase1y = (phase1+subG1[:,:w]@xy)%self.d
                    if phase0y.any()!=0:
                        prob.append(0)
                    else:
                        py = 0
                        for idx in range(pow(self.d,m3)):
                            v = index_to_vector(idx,self.d,m3)
                            phi = np.dot(phase1y,v)%self.d
                            py += pow(omega,phi)*GcT_dict[tuple(v)]
                        py *= pow(self.d,xi-w-m3)
                        prob.append(np.abs(py.real))
                y = np.random.choice(self.d, p = np.array(prob)/px)
                px = prob[y]
                x = np.concatenate([x,np.array([y],dtype=int)])
            res.append(x)
            pxs.append(px)
        return res,pxs



    
                

                    



                