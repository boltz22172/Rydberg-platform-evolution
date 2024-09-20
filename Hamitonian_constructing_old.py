import numpy as np
from scipy.sparse import coo_matrix



'''
Old version of the Hamiltonian constructing part.
We don't use it.
But keep it as a reference.
'''

# Fibonacci function will be useful when determining the size of the Hamitonian
def Fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
def count_zeros_ones(input_list):
    zeros = input_list.count(0)
    ones = input_list.count(1)
    return ones - zeros

def flip(x):
    if x == 1:
        return 0
    elif x == 0:
        return 1
    else:
        raise ValueError("Input must be 0 or 1")
    
I = np.array([[1, 0], [0, 1]])
Sigma_x = np.array([[0, 1], [1, 0]])
Sigma_z = np.array([[1, 0], [0, -1]])
P = np.array([[0, 0], [0, 1]])

def get_Hamiltonian(n, delta, Omega=1):
    """
    Construct the n-site Hamitonian of a PXP model.

    Parameters
    ----------
    n : int
        The size of the system.
    delta : float
        The detuning parameter.
    Omega : float, optional
        The Rabi frequency (default is 1).
    
    Returns
    -------
    Hamiltonian : csc_matrix
        The resulting Hamiltonian matrix in Compressed Sparse Column (CSC) arrays sparse format.

    """
    zero_list = [0] * n
    basis = [zero_list]
    N = Fibonacci(n+2)   # the size of the Hamitonian 

    # use COO format to store the Hamiltonian
    rows = []
    cols = []
    data = []
    
    # This part includes the operation of looking up for elements in the basis and adding new elements to the basis
    # ,which, as suggested by ChatGPT, may be optimized. However, I am not going to do that for now.
    i = 1
    while i ==1 or i!=len(basis):
        vec = basis[i-1]
        for j in range(1,n+1):
            new_vec = vec.copy()
            if j == 1:
                if vec[1] ==0:
                    new_vec[0] = flip(vec[0])
                else:
                    new_vec = [0]
            elif j == n:
                if vec[n-2] == 0:
                    new_vec[n-1] = flip(vec[n-1])
                else:
                    new_vec = [0]
            else:
                if vec[j-2] == 0 and vec[j] == 0:
                    new_vec[j-1] = flip(vec[j-1])
                else:
                    new_vec = [0]
            if new_vec != [0]:
                if new_vec not in basis:
                    basis.append(new_vec)
                position = basis.index(new_vec)
            
                # record the elements of the Hamiltonian
                rows.append(position)
                cols.append(i-1)
                data.append(Omega)
        i = i + 1

    # dealing with the diagonal elements, which means the energy of detuning
    for i in range(N):
        parameter = count_zeros_ones(basis[i]) # number of 1s - number of 0s
        diagonal_value = parameter * delta / 2  # Here, 10 is the number of delta, the detuning
    
        # record the diagonal elements of the Hamiltonian
        rows.append(i)
        cols.append(i)
        data.append(diagonal_value)

    # construct the sparse matrix Hamiltonian
    Hamiltonian_sparse = coo_matrix((data, (rows, cols)), shape=(N, N))
    return Hamiltonian_sparse.tocsc()

def get_Hamiltonian_step(n, delta):
    '''
    Construct the diagonal part of the Hamiltonian in each step of dt.
    '''
    zero_list = [0] * n
    basis = [zero_list]
    N = Fibonacci(n+2)   # the size of the Hamitonian 

    # use COO format to store the Hamiltonian
    rows = []
    cols = []
    data = []
    # dealing with the diagonal elements, which means the energy of detuning
    for i in range(N):
        parameter = count_zeros_ones(basis[i]) # number of 1s - number of 0s
        diagonal_value = parameter * delta / 2  # Here, 10 is the number of delta, the detuning
    
        # record the diagonal elements of the Hamiltonian
        rows.append(i)
        cols.append(i)
        data.append(diagonal_value)

    # construct the sparse matrix Hamiltonian
    Hamiltonian_sparse = coo_matrix((data, (rows, cols)), shape=(N, N))
    return Hamiltonian_sparse.tocsc()

