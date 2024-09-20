import numpy as np
from scipy.sparse import coo_matrix

def flip(bit):
    """Flip a bit (0 -> 1, 1 -> 0)."""
    return 1 - bit

def count_zeros_ones(state, n):
    """Calculate (number of 1s - number of 0s) in the state."""
    ones = bin(state).count('1')
    return ones - (n - ones)

def generate_basis_generator(n):
    """
    Generator to yield all valid basis states for the PXP model using recursion.
    
    Parameters
    ----------
    n : int
        The size of the system.
    
    Yields
    ------
    state : int
        An integer representing a valid basis state.
    """
    # 主要是递归算法。详细的解释可以看git上面的explanation.md
    # 这个算法的好处在于可以省时间，不用一直不停检索。相比于OLD版本，生成的哈密顿量略有不同，但是0000...态和1010...的位置都是开头和末尾，所以问题不大。
    def recurse(state, position):
        if position == n:
            yield state
            return
        # Place a 0
        yield from recurse(state, position + 1)
        # Place a 1 if previous position is 0
        if position == 0 or not (state & (1 << (position - 1))):
            yield from recurse(state | (1 << position), position + 1)
    
    yield from recurse(0, 0)

def get_Hamiltonian_sparse(n, delta, Omega=1):
    """
    Construct the n-site Hamiltonian of a PXP model using generators to handle large n.
    
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
        The resulting Hamiltonian matrix in Compressed Sparse Column (CSC) format.
    """
    # First pass: assign indices to basis states
    state_to_index = {}
    basis_list = []
    for idx, state in enumerate(generate_basis_generator(n)):
        state_to_index[state] = idx
        basis_list.append(state)
    N = len(basis_list)
    
    # Initialize lists for COO format
    rows = []
    cols = []
    data = []
    
    # Second pass: construct Hamiltonian elements
    for i, state in enumerate(basis_list):
        # Diagonal element
        parameter = count_zeros_ones(state, n)
        diagonal_value = parameter * delta / 2
        rows.append(i)
        cols.append(i)
        data.append(diagonal_value)
        
        # Off-diagonal elements: flipping allowed spins
        for j in range(n):
            if ((state >> j) & 1) == 0:
                if (j == 0 or ((state >> (j-1)) & 1) == 0) and \
                   (j == n-1 or ((state >> (j+1)) & 1) == 0):
                    # Flip spin j to 1
                    new_state = state | (1 << j)
                    if new_state in state_to_index:
                        new_index = state_to_index[new_state]
                        rows.append(new_index)
                        cols.append(i)
                        data.append(Omega)
            else:
                # Flip spin j back to 0
                new_state = state & ~(1 << j)
                if new_state in state_to_index:
                    new_index = state_to_index[new_state]
                    rows.append(new_index)
                    cols.append(i)
                    data.append(Omega)
    
    # Construct the sparse Hamiltonian matrix
    Hamiltonian_sparse = coo_matrix((data, (rows, cols)), shape=(N, N))
    return Hamiltonian_sparse.tocsc()

def get_Hamiltonian_sparse_dia(n, delta):
    """
    Construct the n-site Hamiltonian of a PXP model using generators to handle large n.
    Only diagonal elements are stored in the sparse matrix.

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
        The resulting Hamiltonian matrix in Compressed Sparse Column (CSC) format.
    """
    # First pass: assign indices to basis states
    state_to_index = {}
    basis_list = []
    for idx, state in enumerate(generate_basis_generator(n)):
        state_to_index[state] = idx
        basis_list.append(state)
    N = len(basis_list)
    
    # Initialize lists for COO format
    rows = []
    cols = []
    data = []
    
    # Second pass: construct Hamiltonian elements
    for i, state in enumerate(basis_list):
        # Diagonal element
        parameter = count_zeros_ones(state, n)
        diagonal_value = parameter * delta / 2
        rows.append(i)
        cols.append(i)
        data.append(diagonal_value)
    # Construct the sparse Hamiltonian matrix
    Hamiltonian_sparse = coo_matrix((data, (rows, cols)), shape=(N, N))
    return Hamiltonian_sparse.tocsc()
