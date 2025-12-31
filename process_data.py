import numpy as np
import matplotlib.pyplot as plt

# multilayer
X = np.arange(10)
Y = []
for m in [1,2]:
    Ym = []
    for i in X:
        data = np.load(f'0505/multilayer/d=2,n=50,m={m},rand{i}.npy')
        var = np.var(data)
        Ym.append(var)
    Y.append(Ym)
fig = plt.figure(figsize=(8,6))
for m in [1,2]:
    plt.scatter(X,Y[m-1],label=f'm = {m+1}')
plt.xlabel('Number')
plt.ylabel('Variance')
plt.title('d=2,n=50,N=20000,multilayer')
plt.legend()
plt.show()

# XYZ
X = np.arange(10)
Y = []
for i in X:
    data = np.load(f'0505/XYZ/d=2,n=50,XYZ,rand{i}.npy')
    var = np.var(data)
    Y.append(var)
fig = plt.figure(figsize=(8,6))
plt.scatter(X,Y)
plt.xlabel('Number')
plt.ylabel('Variance')
plt.title('d=2,n=50,N=20000,XYZ')
plt.show()
# X = [1000*i for i in range(1,21)]
# Y,Z = [],[]
# i = 10
# data = np.load(f'0325/XYZ/d=2,n=10,XYZ,rand{i}.npy')
# for N in X:
#     tmp = np.random.choice(data, N, replace=False)
#     mean = np.mean(tmp)
#     Y.append(mean)
#     var = np.var(tmp)
#     Z.append(var)
# fig = plt.figure(figsize=(8,6))
# ax1 = fig.add_subplot(111)
# ax1.plot(X,Y, 'o-')
# ax1.set_xlabel('Sample Number')
# ax1.set_ylabel('Mean')
# ax1.set_xticks([2000*i for i in range(1,11)])
# ax2 = ax1.twinx()
# ax2.plot(X,Z, 's--', color='orange')
# ax2.set_ylabel('Variance')
# ax2.set_title('d=2,n=10,XYZ')
# plt.show()

# singlelayer
# X = np.arange(11)
# Y = []
# for j in range(10):
#     Yj = []
#     for h in X:
#         data = np.load(f'0325/singlelayer/d=2,n=10,h={h},XYZ,rand{j}.npy')
#         var = np.var(data)
#         Yj.append(var)
#     Y.append(Yj)
# fig = plt.figure(figsize=(8,6))
# for j in range(10):
#     plt.plot(X,Y[j],'o:')
# # plt.yscale('log')
# plt.xlabel('Number of H')
# plt.ylabel('Variance')
# plt.title('d=2,n=10,N=50000,singlelayer')
# plt.show()
