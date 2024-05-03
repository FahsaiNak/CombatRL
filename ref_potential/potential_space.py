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
def potential(A, B, c1=-0.22, c2=5):
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
            PspaceA[i, j], PspaceB[i, j] = potential(A, B)

    Npos = len(all_combi)**2
    percent = init_percent
    puse = sorted(PspaceA.flatten())[:int(percent*Npos)][-1]
    idxuse = np.where(PspaceA <= puse)
    #print(len(idxuse[0])/Npos)
    while len(idxuse[0])/Npos < percent or percent <= best_percent:
            percent += 0.00001
            puse = sorted(PspaceA.flatten())[:int(percent*Npos)][-1]
            idxuse = np.where(PspaceA <= puse)

    puseA = puse
    #print(puse)
    actual_best = len(idxuse[0])/Npos
    #print(actual_best,len(idxuse[0]) )
    percent_win = sum(PspaceA.ravel()< PspaceB.ravel())/Npos + sum(PspaceA.ravel() == PspaceB.ravel())/Npos
    #print(percent_win)

    percent = init_percent
    puse = sorted(PspaceB.flatten())[:int(percent*Npos)][-1]
    idxuse = np.where(PspaceB <= puse)
    #print(len(idxuse[0])/Npos)
    while len(idxuse[0])/Npos < percent or percent <= best_percent:
            percent += 0.00001
            puse = sorted(PspaceB.flatten())[:int(percent*Npos)][-1]
            idxuse = np.where(PspaceB <= puse)
    puseB = puse

    return puseA, puseB, percent_win, actual_best


N = [4,5,6,7,8]
actual_best = []
for n in N:
    puseA,puseB, percent_win, best = BestPotential(n, 0.009, 0.005)
    actual_best.append(best)
    print("N:",n,"PA: ", puseA, "PB: ",puseB, percent_win, best)

plt.plot(N, actual_best)
plt.show()


N = 6
best_want = [0.009, 0.018, 0.027, 0.036, 0.045, 0.054, 0.063, 0.072, 0.081, 0.09]
actual_best = []
for best in best_want:
    puse, percent_win, best = BestPotential(N, best, 0.005)
    actual_best.append(best)
    print(best, puse, percent_win)

plt.plot(best_want, actual_best)
plt.show()

# plot PspaceA in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = np.arange(len(all_combi))
# y = np.arange(len(all_combi))
# X, Y = np.meshgrid(x, y)
# Z = PspaceA[X, Y]
# ax.plot_surface(X, Y, Z)
# Z2 = PspaceB[X, Y]
# ax.plot_surface(X, Y, Z2)
# plt.show()

# plt.hist(PspaceA.flatten(), bins=100)
# plt.show()
