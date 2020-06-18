from __future__ import print_function
import argparse, math, random
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import torch.backends.cudnn as cudnn

import models, loader, optim
import numpy as np
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Entropy-SGD')
ap = parser.add_argument
ap('-m', help='mnistfc | mnistconv | allcnn | sine', type=str, default='mnistconv')
ap('-b',help='Batch size', type=int, default=128)
ap('-B', help='Max epochs', type=int, default=100)
ap('--lr', help='Learning rate', type=float, default=0.1)
ap('--l2', help='L2', type=float, default=0.0)
ap('-L', help='Langevin iterations', type=int, default=0)
ap('--gamma', help='gamma', type=float, default=1e-4)
ap('--scoping', help='scoping', type=float, default=1e-3)
ap('--noise', help='SGLD noise', type=float, default=1e-4)
ap('-g', help='GPU idx.', type=int, default=0)
ap('-s', help='seed', type=int, default=42)
opt = vars(parser.parse_args())

th.set_num_threads(2)
opt['cuda'] = th.cuda.is_available()
if opt['cuda']:
    opt['g'] = -1
    th.cuda.set_device(opt['g'])
    th.cuda.manual_seed(opt['s'])
    cudnn.benchmark = True
random.seed(opt['s'])
# np.random.seed(opt['s'])
th.manual_seed(opt['s'])
rng = np.random.RandomState(opt['s'])

if 'mnist' in opt['m']:
    opt['dataset'] = 'mnist'
elif 'allcnn' in opt['m']:
    opt['dataset'] = 'cifar10'
elif 'sine' in opt['m']:
    x_all = np.linspace(-5, 5, 50)[:, None]
else:
    assert False, "Unknown opt['m']: " + opt['m']

def gen_task():
    "Generate regreesion problem"
    phase = rng.uniform(low=0, high=2*np.pi) 
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x: np.sin(x+phase)*ampl
    return f_randomsine

model = nn.Sequential(
    nn.Linear(1,64),
    nn.Tanh(),
    nn.Linear(64,64),
    nn.Tanh(),
    nn.Tanh(64, 1),
)

def to_torch(x):
    return Variable(torch.Tensor(x))

def train_on_bath(x, y):
    x = to_torch(x)
    y = to_torch(y)
    model.zero_grad()
    ypred = model(x)
    loss = (ypred-y).pow(2).mean()

optimizer = optim.EntropySGD(model.parameters(),
        config = dict(lr=opt['lr'], momentum=0.9, nesterov=True, weight_decay=opt['l2'],
        L=opt['L'], eps=opt['noise'], g0=opt['gamma'], g1=opt['scoping']))

def experiment(plot=True):
    f_plot = gen_task()
    xtrain_plot = x_all[rng.choice(len(x_all), size=n_train)]
    
    n_train = opt['b']

    for iteration in range(opt['B']):
        f = gen_task()
        y_all = f(x_all)
        inds = rng.permutation(len(x_all))
        train_ind = inds[:-1*n_train]
        val_ind = inds[-1, n_train:]
if __name__ == '__main__':
    experiment()
