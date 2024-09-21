from scipy.sparse.linalg import splu
from scipy.sparse import identity
from scipy.sparse.linalg import expm_multiply
import numpy as np
import math
from Hamitonian_constructing import get_Hamiltonian_sparse,get_Hamiltonian_sparse_dia

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

# The function that will be used to evolve the state
def evolve(n,T,delta0,dt,stepper):
    '''
    Do the state evolution for a given time T with a given time step dt
    Start from the initial state |0000...> and evolve the state with the Hamiltonian.
    Here we let the intensity of laser to be a constant (Omega = 1), 
    and the detuning to be a linear function of time (delta = delta0 - 2 * delta0 * t / T)

    Parameters:
    ----------
    n: int
        The size of system
    T: float
        The total time of evolution
    delta0: float
        The absolute value of the detuning at t=0
    dt: float
        The time step
    stepper: function
        The function that will be used to evolve the state

    Returns:
    -------
    state: array
        The state of the system after evolution
    
    '''
    initial_state = np.zeros(Fibonacci(n+2))
    initial_state[0] = 1    # The state |0000..> represented as a vector in matrix form is indeed [1,0,0,...]
    numsteps = math.floor(T / dt)   
    state = initial_state
    H = get_Hamiltonian_sparse(n, delta0)   # The hamitonian at t=0
    H_step  = get_Hamiltonian_sparse_dia(n,- 2 * delta0 * dt / T)   # The change of the Hamiltonian in each step of dt
    for i in range(numsteps):
        times = i * dt
        state = stepper(state, dt, times, H, H_step)
    return state

def evolve_1_site(T, delta0, dt, stepper):
    '''
    Do the time evolution to **a single site** system with a given time step dt

    Parameters:
    ----------
    T: float
        The total time of evolution
    delta0: float
        The absolute value of the detuning at t=0
    dt: float
        The time step
    stepper: function
        The function that will be used to evolve the state
    
    Returns:
    -------
    state: array
        The state of the system after evolution
    '''
    # alpha = 2*delta0/T, it's the slope of the change of \delta (The detuning)
    initial_state = np.zeros(2)
    initial_state[1] = 1
    numsteps = math.floor(T / dt)
    state = initial_state
    H = np.array([[delta0, 1], [1, -delta0]])  # The Hamiltonian at t=0
    H_step = np.array([[-2*delta0*dt/T, 0], [0, 2*delta0*dt/T]])  # The change of the Hamiltonian in each step of dt
    for i in range(numsteps):
        times = i * dt
        state = stepper(state, dt, times, H, H_step)
    return state

#-------------------------------------------------------------

## Here are the functions that will be used to evolve the state
## They are: euler_step, runge_kutta_step, unitary_euler_step, exponential_step

def euler_step(state, dt, times, H, H_step):
    H_t = H + times/dt * H_step
    return state + (-1j * dt * H_t @ state)

def runge_kutta_step(state, dt, times, H, H_step):
    H_t = H + times/dt * H_step
    H_half = H + (times/dt + 1/2)*H_step
    H_full = H + (times/dt + 1)*H_step
    k1 = -1j  * H_t @ state
    k2 = -1j  * H_half @ (state + k1*dt / 2)
    k3 = -1j  * H_half @ (state + k2*dt / 2)
    k4 = -1j  * H_full @ (state + k3*dt)
    return state + (k1 + 2 * k2 + 2 * k3 + k4)*dt / 6

def unitary_euler_step(state, dt, times, H, H_step):
    H_t = H + times/dt * H_step
    I = identity(H_t.shape[0], format='csc')  # Use the appropriate sparse identity matrix
    M_p = I + 1j * dt/2 * H_t
    M_m = I - 1j * dt/2 * H_t
    right_vec = M_m @ state
    lu = splu(M_p)
    return lu.solve(right_vec)

def exponential_step(state, dt, times, H, H_step):
    H_t = H + times/dt * H_step
    return expm_multiply(-1j*dt*H_t,state)


#-------------------------------------------------------------

def Possibility_of_ideal_state(n,T,delta0,dt,stepper):
    '''
    Calculate the possibility of the system to be in the ideal state |1010...> after evolution

    Parameters:
    ----------
    n: int
        The size of system
    T: float
        The total time of evolution
    delta0: float
        The absolute value of the detuning at t=0
    dt: float
        The time step
    stepper: function
        The function that will be used to evolve the state

    Returns:
    -------
    The possibility of the system to be in the ideal state |1010...> after evolution

    '''
    final_state = evolve(n,T,delta0,dt,stepper)
    ideal_state = np.zeros(Fibonacci(n+2))
    ideal_state[-1] = 1
    return abs(ideal_state.conj().T @ final_state) / np.linalg.norm(final_state)

def Possibility_of_ideal_state_1_site(T, delta0, dt, stepper):
    '''
    Calculate the possibility of the system to be in the ideal state |1010...> after evolution, for a single site system

    Parameters:
    ----------
    n: int
        The size of system
    T: float
        The total time of evolution
    delta0: float
        The absolute value of the detuning at t=0
    dt: float
        The time step
    stepper: function
        The function that will be used to evolve the state

    Returns:
    -------
    The possibility of the system to be in the ideal state |1010...> after evolution
    '''
    final_state = evolve_1_site(T, delta0, dt, stepper)
    ideal_state = np.zeros(2)
    ideal_state[0] = 1
    return abs(ideal_state.conj().T @ final_state) / np.linalg.norm(final_state)