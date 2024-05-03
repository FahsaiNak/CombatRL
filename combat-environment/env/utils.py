import numpy as np


#Function to calcualte potential for A and B
def calculate_potential(A, B):
    n = len(A)
    pavg = 0.6 * n**2
    p_sd = 0.15 * n**2 
    pAi = ((A-B).sum())**2
    pA = 1/(p_sd*np.sqrt(2*np.pi))*np.exp(1/2*(pAi-pavg)**2/p_sd**2)
    return pA, n**2-pA