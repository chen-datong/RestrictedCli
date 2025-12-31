import numpy as np

def commutator_t(x,y,d,m):
    # in z1,z2,...,zn,x1,x2,....,xn basis
    Jt = np.zeros((2*m,2*m),dtype=int)
    Jt[:m,m:2*m] = np.identity(m,dtype=int)
    Jt[m:2*m,:m] = -np.identity(m,dtype=int)
    return (x@Jt@y)%d

def divide(a,b,d):
    # convert a/b to c \in F_d
    a = a%d
    b = b%d
    for i in range(d):
        if (i*d+a)%b == 0:
            return int((i*d+a)/b)

def compute_divideList(d):
    # divideList[num][dem]
    divideList = np.zeros((d,d), dtype=int)
    for i in range(d):
        for j in range(1,d):
            divideList[i,j] = divide(i,j,d)
    return divideList        

def index_to_vector(idx,d,l):
    v = np.zeros(l,dtype=int)
    r = idx
    for j in range(l):
        v[j] = r//(pow(d,l-1-j))
        r = r%(pow(d,l-1-j))
    return v

def Zassenhaus(U,W,d):
    # U = [u1;u2;...;un]
    # W = [w1;w2;...;wk]
    n,m = np.shape(U)
    k,_ = np.shape(W)
    M = np.concatenate([np.concatenate([U,U],axis=1), np.concatenate([W,np.zeros((k,m),dtype=int)],axis=1)])
    echM = echelonMatrix(M,d)
    echM.echelon()
    cut2 = echM.rk
    cut1 = cut2
    for i in range(cut2-1,-1,-1):
        if echM.firstSite[i]>=m: cut1 -= 1
        else:   break
    C = echM.M[0:cut1,0:m] # basis of U+W
    D = echM.M[cut1:cut2,m:2*m] # basis of U \cap W
    return C,D



primitiveElements = {3:2, 5:2, 7:3, 11:2, 13:2, 17:3, 19:2, 23:5, 29:2, 31:3, 37:2, 41:6, 43:3, 47:5}

def gadget(d,order): # d!=2
    if d == 3:
        ga = np.array([0,1,8])
        return ga
    else:
        mu = primitiveElements[d]
        ga = np.array([-k**3 for k in range(d)],dtype=int)
        return (pow(mu,order)*ga)%d

def compute_WT(d):
    if d==2:
        # key: (Z,X)
        ga = np.array([0,7], dtype=int)
        zeta=np.exp(2*np.pi*1j/8)
        WT_dict = {}
        WT_dict[(0,0)] = 1
        WT_dict[(1,0)] = 0
        WT_dict[(0,1)] = 1/2*(pow(zeta,1)+pow(zeta,-1))
        WT_dict[(1,1)] = 1j/2*(pow(zeta,1)-pow(zeta,-1))
    else:
        # key: (Z,X,j)
        omega = np.exp(2*np.pi*1j/9) if d==3 else np.exp(2*np.pi*1j/d)
        c = 3 if d==3 else 1
        divideList = compute_divideList(d)
        WT_dict = {}
        for j in range(3):
            if d%3 != 1 and j>0:   break
            for z in range(d):
                for x in range(d):
                    Tr = gadget(d,j)
                    Tr = np.array([Tr[(i-x)%d] for i in range(d)])
                    Tr = np.array([(Tr[i]+c*z*i)%(c*d) for i in range(d)])               
                    WT_dict[(z,x,j)] = 1/d*pow(omega,-c*divideList[(z*x)%d,2])*sum(pow(omega,Tr-gadget(d,j)))
    return WT_dict

class echelonMatrix:
    def __init__(self,M,d):
        self.M = M
        self.m, self.n = np.shape(self.M) # n>=m
        self.rk = None
        self.firstSite = []
        self.d = d
        self.divideList = compute_divideList(self.d)
        self.Sleft = None
        self.Sright = None
        self.emptyRows = []
        

    def echelon(self,row_exchange=True):
        i,j = 0,0
        Sleft, Sright = np.identity(self.m,dtype=int), np.identity(self.n,dtype=int)
        while i<self.m and j<self.n:
            if self.M[i,j]==0:
                if row_exchange:
                    S = np.identity(self.m,dtype=int)
                    for k in range(i+1,self.m):
                        if self.M[k,j]!=0:
                            S[i,i],S[k,k],S[i,k],S[k,i] = 0,0,1,1
                            break
                    else:
                        j+=1
                        continue
                    self.M = S@self.M
                    Sleft = S@Sleft
                else:
                    S = np.identity(self.n,dtype=int)
                    for l in range(j+1,self.n):
                        if self.M[i,l]!=0:
                            S[j,j],S[l,l],S[j,l],S[l,j] = 0,0,1,1
                            break
                    else:
                        self.emptyRows.append(i)
                        i+=1
                        continue
                    self.M = self.M@S
                    Sright = Sright@S

            self.firstSite.append(j)
            # forward elimination
            S = np.identity(self.m,dtype=int)
            S[i,i] = self.divideList[1,self.M[i,j]%self.d]
            if i<self.m-1:
                S[i+1:,i] = np.array([-self.divideList[self.M[k,j]%self.d,self.M[i,j]%self.d] for k in range(i+1,self.m)])
            self.M = (S@self.M)%self.d
            Sleft = (S@Sleft)%self.d
            i += 1
            j += 1
        self.Sleft = Sleft
        self.Sright = Sright
        self.rk = i if row_exchange else i-len(self.emptyRows)
        

    def reduced_echelon(self):
        self.echelon()

        # back substitution
        for i in range(self.rk-1,0,-1):
            S = np.identity((self.m),dtype=int)
            S[:i,i] = -self.M[:i,self.firstSite[i]]
            self.M = S@self.M
            self.Sleft = S@self.Sleft


    def inverse(self): 
        # m=n, full rank
        Minv = np.identity((self.n),dtype=int)
        for i in range(self.n): 
            # 0 division
            if self.M[i][i]==0: 
                S = np.identity((self.n),dtype=int)
                for k in range(i+1,self.n):
                    if self.M[k][i]!=0:
                        S[i,i],S[k,k],S[i,k],S[k,i] = 0,0,1,1
                        break
                self.M = S@self.M
                Minv = S@Minv

            # forward elimination
            S = np.identity((self.n),dtype=int)
            S[i,i] = self.divideList[1,self.M[i,i]]
            if i<self.n-1:
                S[i+1:,i] = np.array([-self.divideList[self.M[j,i],self.M[i,i]] for j in range(i+1,self.n)])
            self.M = (S@self.M)%self.d
            Minv = (S@Minv)%self.d

        # back substitution
        for i in range(self.n-1,0,-1):
            S = np.identity((self.n),dtype=int)
            S[:i,i] = -self.M[:i,i]
            self.M = (S@self.M)%self.d
            Minv = (S@Minv)%self.d
        return Minv

def rowsum(u1,u2,n):
    # in tableau representation
    # u1, u2 in T
    # phase must be either 1 or -1
    g = 0
    for j in range(n):
        if u1[j]==0 and u1[n+j]==0: g += 0
        if u1[j]==1 and u1[n+j]==1: g += u2[j]-u2[n+j]
        if u1[j]==0 and u1[n+j]==1: g += u2[j]*(2*u2[n+j]-1)
        if u1[j]==1 and u1[n+j]==0: g += u2[n+j]*(1-2*u2[j])
    return g