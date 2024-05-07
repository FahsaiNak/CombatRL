import numpy as np


#Function to calcualte potential for A and B
def calculate_potential_v0(A, B):
    n = len(A)
    pavg = 0.6 * n**2
    p_sd = 0.15 * n**2 
    pAi = ((A-B).sum())**2
    pA = 1/(p_sd*np.sqrt(2*np.pi))*np.exp(1/2*(pAi-pavg)**2/p_sd**2)
    return pA, n**2-pA

# Create all combinations of given 1s and 0s
def create_combinations(n, k):
    if k == 0:
        return [np.zeros(n, dtype=int)]
    if n == k:
        return [np.ones(n, dtype=int)]
    return [np.concatenate(([0], x)) for x in create_combinations(n - 1, k)] + [np.concatenate(([1], x)) for x in create_combinations(n - 1, k - 1)]

#Function to calcualte potential for A and B
def calculate_potential(A, B, c1=-0.22, c2=5):
    n = len(A)
    Pa = np.exp(c1*(n-sum(A)))*np.sin(c2*sum(A)/n*np.pi)*np.cos(sum(B)/n*np.pi)
    Pb = np.exp(c1*sum(B))*np.sin(c2*sum(B)/n*np.pi)*np.cos(sum(A)/n*np.pi)
    return Pa, Pb

def BestPotential(N, best_percent = 0.009, init_percent = 0.005):
    
    all_combi = []
    for i in range(N+1):
        all_combi.extend(create_combinations(N, i))

    PspaceA = np.zeros((len(all_combi), len(all_combi)))
    PspaceB = np.zeros((len(all_combi), len(all_combi)))
    for i, A in enumerate(all_combi):
        for j, B in enumerate(all_combi):
            PspaceA[i, j], PspaceB[i, j] = calculate_potential(A, B)

    Npos = len(all_combi)**2
    percent = init_percent
    puse = sorted(PspaceA.flatten())[:int(percent*Npos)][-1]
    idxuse = np.where(PspaceA <= puse)
    while len(idxuse[0])/Npos < percent or percent <= best_percent:
            percent += 0.00001
            puse = sorted(PspaceA.flatten())[:int(percent*Npos)][-1]
            idxuse = np.where(PspaceA <= puse)

    puseA = puse
    actual_best = len(idxuse[0])/Npos
    percent_win = sum(PspaceA.ravel()< PspaceB.ravel())/Npos + sum(PspaceA.ravel() == PspaceB.ravel())/Npos

    percent = init_percent
    puse = sorted(PspaceB.flatten())[:int(percent*Npos)][-1]
    idxuse = np.where(PspaceB <= puse)
    while len(idxuse[0])/Npos < percent or percent <= best_percent:
            percent += 0.00001
            puse = sorted(PspaceB.flatten())[:int(percent*Npos)][-1]
            idxuse = np.where(PspaceB <= puse)
    puseB = puse

    return puseA, puseB, percent_win, actual_best