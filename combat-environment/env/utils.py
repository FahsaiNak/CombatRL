import numpy as np

def create_combinations(n, k):
    if k == 0:
        return [np.zeros(n, dtype=int)]
    if n == k:
        return [np.ones(n, dtype=int)]
    return [np.concatenate(([0], x)) for x in create_combinations(n - 1, k)] + [np.concatenate(([1], x)) for x in create_combinations(n - 1, k - 1)]

#Function to calcualte potential for A and B
def potential(A, B):
    n = len(A)
    pavg = 0.6 * n**2
    p_sd = 0.15 * n**2 
    pAi = ((A-B).sum())**2
    pA = 1/(p_sd*np.sqrt(2*np.pi))*np.exp(1/2*(pAi-pavg)**2/p_sd**2)
    return pA, n**2-pA


    