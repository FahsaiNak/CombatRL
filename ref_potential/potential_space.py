import matplotlib.pyplot as plt
import numpy as np


# Create all combinations of given 1s and 0s
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


N = 10
all_combi = []
for i in range(N+1):
    all_combi.extend(create_combinations(N, i))



k = 2
PspaceA = np.zeros((len(all_combi), len(all_combi)))
PspaceB = np.zeros((len(all_combi), len(all_combi)))
for i, A in enumerate(all_combi):
    for j, B in enumerate(all_combi):
            pAi, pBj = potential(A, B)
            if np.count_nonzero(A) == np.count_nonzero(B) and np.count_nonzero(A) != k:
                PspaceA[i, j] = pAi+np.count_nonzero(B)
                PspaceB[i, j] = pBj+np.count_nonzero(B)
            elif np.count_nonzero(A) == np.count_nonzero(B) and np.count_nonzero(A) == k:
                PspaceA[i, j] = 0
                PspaceB[i, j] = 100
            else:
                PspaceA[i, j] = pAi
                PspaceB[i, j] = pBj

print(np.count_nonzero(PspaceA))

# plot PspaceA in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(len(all_combi))
y = np.arange(len(all_combi))
X, Y = np.meshgrid(x, y)
Z = PspaceA[X, Y]
ax.plot_surface(X, Y, Z)
# Z2 = PspaceB[X, Y]
# ax.plot_surface(X, Y, Z2)
plt.show()

plt.hist(PspaceA.flatten(), bins=100)
plt.show()