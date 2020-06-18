import numpy as np
import matplotlib.pyplot as plt

err_maml = np.loadtxt('MAML.txt')
err_reptile = np.loadtxt('Reptile.txt')
err_entropy_ent = np.loadtxt('Entropy-sgd.txt')
err_entropy_sgd = np.loadtxt('Entropy-SGD.txt')
err_mlap = np.loadtxt('MLAP.txt')
err_sketch = np.loadtxt('sketch.txt')
err_maml_ent = np.loadtxt('MAML-ent.txt')
err_ent = np.loadtxt('ent.txt')
err_sketch_ent = np.loadtxt('sketch-ent.txt')
n_plot = err_maml.shape[0]
n_epoch = err_maml.shape[1]

x = range(0, 10*(n_epoch),10)
# print(list(x))

for i in range(n_plot):
    plt.cla()
    plt.plot(x,err_maml[i,:], label='MAML')
    plt.plot(x, err_maml_ent[i,:], label='MAML + Entropy-SGD')
    plt.plot(x,err_reptile[i,:], label='Reptile')
    plt.plot(x,err_mlap[i,:], label='MLAP')
    plt.plot(x,err_entropy_sgd[i,:], label='Entropy-SGD + SGD')
    # plt.plot(x,err_ent[i,:],label='ent')
    plt.plot(x,err_entropy_ent[i,:], label='Entropy-SGD + Entropy-SGD')
    plt.plot(x,err_sketch[i,:],'-', label='Sketch')
    plt.plot(x,err_sketch_ent[i,:], label='Sketch + Entropy-SGD')
    plt.ylim(0,8)
    plt.legend(loc='upper right')
    name = str(i) + 'ss.jpg'
    plt.savefig(name)
    plt.close()
