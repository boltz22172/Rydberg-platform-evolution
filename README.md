# Rydberg-platform-evolution
Building up the Hamitonian of a Rydberg platform, and calculate its time evolution. Various numerical techniques may be used.

The motivation for this project comes from several papers by the Lukin group at Harvard University on the Rydberg atom platform. Here is some relevant references:
https://www.nature.com/articles/s41586-021-03582-4 

According to the paper, researchers chosed to use **adiabatic evolution** to prepare the checkerboard phase. They use laser to act with Rydberg atoms and keep the Rabi frequncy $\Omega$ to be a constant. Then, gradually modulate the detuning $\Delta$, so that the state will stay in the ground state, evolving to the checkerboard phase. Since we want this process to be adiabatic to achieve high accuracy, the time of this evolution will be long.

What we want to optimize is to come up with a new protocal called **Sweeping protocal**, which we hope will accelerate this whole procedure and get even better accuracy. The main idea of this is to operate atoms one by one.

Our work will contain both Numerical Analysis and theoretical analysis.
