import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as LAs
import Sub180221 as Sub
import math,copy

def normalize_max(matrix):
    max_val = np.max(np.abs(matrix))
    return matrix / max_val if max_val != 0 else matrix

def contains_inf_or_nan(arr):
    return np.any(np.isnan(arr)) or np.any(np.isinf(arr))

### Initialize a random Mps/初始化MPS，初始化的结果是随机的
def InitMps(Ns, Dp, Ds):
    '''
    Given the system size, initialize an Mps in Canonical form. Ds is the truncation index of the virtual dimension.
    -------------------
    Ns: int, system size
    Dp: int, physical dimension, default is 2
    Ds: int, truncation index of the virtual dimension
    -------------------
    return
    T: list, Mps
    '''
    T = [None] * Ns
    
    # Randomly initialize Mps
    for i in range(Ns):
        Dl = min(Dp**i, Dp**(Ns-i), Ds)
        Dr = min(Dp**(i+1), Dp**(Ns-i-1), Ds)
        T[i] = np.random.rand(Dl, Dp, Dr)
        
    # Canonical form
    U = np.eye(np.shape(T[-1])[-1])
    for i in range(Ns-1, 0, -1):
        U, T[i] = Sub.Mps_LQP(T[i], U)    #####
        U = normalize_max(U)      # Otherwise, L may become very large, so large that the system considers it as inf
        
    return T

def IniH(Mpo,T):
    Ns = len(T)  # We might want to take multiples of 12 in our specific case
    Dmpo = np.shape(Mpo[0])[0]
    HL = [None]*Ns
    HR = [None]*Ns
    HL[0] = np.zeros((1,Dmpo,1))
    HL[0][0,0,0] = 1.0
    HR[-1] = np.zeros((1,Dmpo,1))
    HR[-1][0,-1,0] = 1.0
    
    for i in range(Ns-1,0,-1):  # Don't get i = 0
        HR[i-1] = Sub.NCon([HR[i],T[i],Mpo[i%6],np.conj(T[i])],[[1,3,5],[-1,2,1],[-2,2,3,4],[-3,4,5]])
    
    return HL,HR

def OptTsite(Mpo_single,HL,HR,T,Method=0):
    DT = np.shape(T)   # What is the shape of T
    Dl = np.prod(DT)   # Dl = D1*D2*...*DN
    
    if Method == 0:
        A = Sub.NCon([HL,Mpo_single,HR],[[-1,1,-4],[1,-5,2,-2],[-6,2,-3]])
        A = Sub.Group(A,[[0,1,2],[3,4,5]])
        Eig,V = LAs.eigsh(A,k=1,which = 'SA')
        T = np.reshape(V,DT)
        
    if Method == 1:
        def UpdateV(V):
            V = np.reshape(V,DT)
            V = Sub.NCon([HL,V,Mpo_single,HR],[[-1,3,1],[1,2,4],[3,2,5,-2],[4,5,-3]])
            V = np.reshape(V,Dl)
            return V
        
        V0 = np.reshape(T,Dl)
        MV = LAs.LinearOperator((Dl,Dl),matvec=UpdateV)
        Eig,V = LAs.eigsh(MV,k=1,which='SA',v0=V0)
		# print(Eig)
        T = np.reshape(V,DT)
        Eig = np.real(Eig)
    
    return T,Eig

def OptT(Mpo,HL,HR,T):
    '''
    Sweep back and forth to optimze the Mps
    -------------------
    Mpo: list, Mpo (len = 6)
    HL = initial_HL
    HR = initial_HR
    T = initial_Mps
    -------------------
    return
    T: list, optimized Mps
    EEE: float, energy per site
    '''
    Ns = len(T)
    Eng0 = np.zeros(Ns)
    Eng1 = np.zeros(Ns)
    for r in range(100):   # The number of sweeps
        if r == 99:
            print('Reaching the limit!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #print('-------------------------------------------')
        #print('Sweep:',r)
        for i in range(Ns-1):   # Sweep to the right
            T[i],Eng1[i] = OptTsite(Mpo[i%6],HL[i],HR[i],T[i],Method=1)
            T[i],U = Sub.Mps_QR0P(T[i])
            HL[i+1] = Sub.NCon([HL[i],np.conj(T[i]),Mpo[i%6],T[i]],[[1,3,5],[1,2,-1],[3,4,-2,2],[5,4,-3]])
            T[i+1] = np.tensordot(U,T[i+1],(1,0))
            
        for i in range(Ns-1,0,-1):
            T[i],Eng1[i] = OptTsite(Mpo[i%6],HL[i],HR[i],T[i],Method=1)
            U,T[i] = Sub.Mps_LQ0P(T[i])
            HR[i-1] = Sub.NCon([HR[i],T[i],Mpo[i%6],np.conj(T[i])],[[1,3,5],[-1,2,1],[-2,2,3,4],[-3,4,5]])
            T[i-1] = np.tensordot(T[i-1],U,(2,0))
        #print(Eng1)
        if abs(Eng1[1]-Eng0[1]) < 1.0e-8:      # Stop sweeping if the energy converges, here we choose e-8 as the threshold
            break
        Eng0 = copy.copy(Eng1)
    
    
    #print('Eigenvalue per site:',Eng1 /float(Ns))
    EEE = Eng1[1] /float(Ns)
    return T,EEE

def canonicalize_mps(T,N):
    '''
    Given an Mps, move the center site to the Nth site and mix-canonicalize the Mps
    '''
    # N is the index of our center site
    # T is left-canonicalized
    T_copy = copy.deepcopy(T)
    Ns = len(T)
    for i in range(N): # Move the center site to the right, updating until N-1
        T_copy[i],U = Sub.Mps_QR0P(T_copy[i])
        if i==N-1:
            Values = U
        T_copy[i+1] = np.tensordot(U,T_copy[1+i],(1,0))
        
    return T_copy,T_copy[N]

def calculate_sigmaz(T,N):
    '''
    Calculate the expectation value of the Z operator at the Nth site
    '''
    opt_T,site_N = canonicalize_mps(T,N)
    Z = np.array([[1,0],[0,-1]])
    Z_value = Sub.NCon([site_N,Z,np.conj(site_N)],[[1,2,3],[2,4],[1,4,3]])
    return Z_value

def Entropy_calculation(T,N):  
    '''
    Calculate the entropy of the Nth site
    '''
    Ns = len(T)
    T = copy.deepcopy(T)
    for i in range(N):
        T[i],U = Sub.Mps_QR0P(T[i])
        T[i+1] = np.tensordot(U,T[1+i],(1,0))
    
    T[N-1] = Sub.Group(T[N-1],[[0,1],[2]])
    T[N]   = Sub.Group(T[N],[[0],[1,2]])
    Mat = Sub.NCon([T[N-1],T[N]],[[-1,1],[1,-2]])
    UU,SS,Vh = np.linalg.svd(Mat)
    Values = SS
    evals = np.square(Values)
    evals = evals[evals > 0]
    
    return -np.sum(evals * np.log(evals))

def Entropy_calculation2(T,N):   # Another way to calculate the entropy, actually the same as the previous one
    Ns = len(T)
    for i in range(N):
        T[i],U = Sub.Mps_QR0P(T[i])
        if i==N-1:
            Values = U
        T[i+1] = np.tensordot(U,T[1+i],(1,0))
    Values = Values@Values.T.conj()
    evals = np.linalg.eigvalsh(Values)
    evals = evals[evals > 0]
    
    return -np.sum(evals * np.log(evals))
