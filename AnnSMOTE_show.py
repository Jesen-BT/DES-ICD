import matplotlib.pyplot as plt
import numpy as np
from DES import *
import math
from imblearn.over_sampling import SMOTE

def gd(x, mu=0, sigma=1):
    left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma))
    right = np.exp(-(x - mu)**2 / (2 * sigma))
    return left * right

xlim = np.arange(-4, 5, 0.1)
concept1 = gd(xlim, 0, 1)
concept2 = gd(xlim, 2, 1)


maj_x1, maj_y1 = np.random.multivariate_normal([2, -2], [[1, 0], [0, 1]], 1000).T
maj_x2, maj_y2 = np.random.multivariate_normal([2, -2], [[1, 0], [0, 1]], 1000).T

min_x1, min_y1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100).T
min_x2, min_y2 = np.random.multivariate_normal([0, 2], [[1, 0], [0, 1]], 100).T

min_ox, min_oy = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 50).T
min_nx, min_ny = np.random.multivariate_normal([0, 2], [[1, 0], [0, 1]], 50).T
block2_x = np.append(min_ox, min_nx)
block2_y = np.append(min_oy, min_ny)

#两种概念分布
plt.figure(figsize=(6,3))
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
plt.plot(xlim, concept2, color='coral', label='current concept')
plt.plot(xlim, concept1, color='cyan',  label='previous concept')
ax = plt.gca()
ax.spines['right'].set_color('none')#取消右侧边框
ax.spines['top'].set_color('none')#取消上方边框
plt.legend(loc=0,ncol=1)
plt.savefig('fig/1.jpg', dpi=400)
#两个数据块分布
plt.figure(figsize=(6, 3))
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
plt.hist(min_y1, 30, histtype='stepfilled', orientation='vertical', color='cyan', density=True,label='previous')
plt.hist(block2_y, 30, histtype='stepfilled', orientation='vertical', color='coral', density=True, label='current')
ax = plt.gca()
ax.spines['right'].set_color('none')#取消右侧边框
ax.spines['top'].set_color('none')#取消上方边框
plt.legend(loc=0,ncol=1)
plt.savefig('fig/2.jpg', dpi=400)

fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

main_ax.plot(maj_x2, maj_y2, 'ok', markersize=3, alpha=0.2,color='cyan')
main_ax.plot(block2_x, block2_y, 'ok', markersize=3, alpha=0.2,color='coral')
x_hist.hist(block2_x, 30, histtype='stepfilled',density=True,
             orientation='vertical', color='coral')
x_hist.invert_yaxis()
y_hist.hist(block2_y, 30, histtype='stepfilled',density=True,
             orientation='horizontal', color='coral')
y_hist.plot(concept2, xlim, color='coral')
y_hist.invert_xaxis()
fig.savefig('fig/3.jpg', dpi=600)

refX = np.append(min_x1, maj_x1)
refY = np.append(min_y1, maj_y1)
refB = np.zeros((len(refX), 2))
refBY = np.ones(len(refY))

B_X = np.append(block2_x, maj_x2)
B_Y = np.append(block2_y, maj_y2)
B = np.zeros((len(B_X), 2))
for i in range(len(B_Y)):
    B[i][0] = B_X[i]
    B[i][1] = B_Y[i]
    refB[i][0] = refX[i]
    refB[i][1] = refY[i]
BY = np.ones(len(B_Y))
for i in range(len(block2_y)):
    BY[i] = 0
    refBY[i] = 0

AKS_B, AKS_BY = AKS(refB=refB, refBy=refBY, newB=B, newBy=BY)

AKS_B_x = []
AKS_B_y = []

for i in range(len(AKS_BY)):
    if AKS_BY[i] == 0:
        AKS_B_x.append(AKS_B[i][0])
        AKS_B_y.append(AKS_B[i][1])

fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

main_ax.plot(maj_x2, maj_y2, 'ok', markersize=3, alpha=0.2,color='cyan')
main_ax.plot(AKS_B_x, AKS_B_y, 'ok', markersize=3, alpha=0.2,color='red')
x_hist.hist(AKS_B_x, 30, histtype='stepfilled',density=True,
             orientation='vertical', color='lightcoral')
x_hist.invert_yaxis()
y_hist.hist(AKS_B_y, 30, histtype='stepfilled',density=True,
             orientation='horizontal', color='lightcoral')
y_hist.plot(concept2, xlim, color='coral')
y_hist.invert_xaxis()
fig.savefig('fig/4.jpg', dpi=400)

sm = SMOTE(random_state=42)
smB, smBY = sm.fit_resample(X=B, y=BY)
smB_x=[]
smB_y=[]
for i in range(len(smBY)):
    if smBY[i] == 0:
        smB_x.append(smB[i][0])
        smB_y.append(smB[i][1])

fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

main_ax.plot(maj_x2, maj_y2, 'ok', markersize=3, alpha=0.2,color='cyan', label='samples of majority class')
main_ax.plot(smB_x, smB_y, 'ok', markersize=3, alpha=0.2,color='red', label='Synthetic samples')
main_ax.plot(block2_x, block2_y, 'ok',markersize=3, alpha=0.2,color='coral',label='samples of minority class')
x_hist.hist(smB_x, 30, histtype='stepfilled',
             orientation='vertical', density=True, color='lightcoral')
x_hist.invert_yaxis()
y_hist.hist(smB_y, 30, histtype='stepfilled',density=True,
             orientation='horizontal', color='lightcoral')
y_hist.plot(concept2, xlim, color='coral')
y_hist.invert_xaxis()
main_ax.legend(loc=0,ncol=1,fontsize=7)
fig.savefig('fig/5.jpg', dpi=400)

plt.show()