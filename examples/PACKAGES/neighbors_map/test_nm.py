import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('test_nm.dump',skiprows=9)[:,4:]
imsize = int(np.sqrt(np.shape(data)[1]))
N = 4
M=len(data)
ids = np.random.randint(0,M,size=N*N)
print(ids)

fig,axes = plt.subplots(N,N,figsize=(16,16))
axes = axes.flatten()
for ax, idx in zip(axes, ids):
    dataloc = np.reshape(data[idx,:],(imsize,imsize))
    ax.imshow(dataloc)#,vmin=0.,vmax=0.01)
plt.tight_layout()    
plt.savefig('test_nm_plot.png')
