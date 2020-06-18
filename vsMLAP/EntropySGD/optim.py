from torch.optim import Optimizer
from copy import deepcopy
import numpy as np

class EntropySGD(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(llr=0.1, lr=0.1, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=20, eps=1e-4, g0=1e-2, g1=1e-1)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(EntropySGD, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, prm=None, model=None, criterion=None, optimizer=None, iterators=None, data_loaders=None, i_epoch=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for Entropy-SGD, model and criterion'
        mf,_,_ = closure(prm, model, criterion, iterators, data_loaders, optimizer)

        c = self.config
        if prm.lr_schedule:
            lr = self.param_groups[0]['lr']
        else:
            lr = c['lr']
        llr = c['llr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = int(c['L'])
        eps = c['eps']
        g0 = c['g0']
        g1 = c['g1']

        params = self.param_groups[0]['params']

        state = self.state
        # initialize
        if not 't' in state:
            state['t'] = 0
            state['wc'], state['mdw'] = [], []
            for w in params:
                state['wc'].append(deepcopy(w.data))
                state['mdw'].append(deepcopy(w.grad.data))

            state['langevin'] = dict(mw=deepcopy(state['wc']),
                                    mdw=deepcopy(state['mdw']),
                                    eta=deepcopy(state['mdw']),
                                    llr = llr,
                                    beta1 = 0.75)
        # state['t'] = i_epoch
        lp = state['langevin']
        for i,w in enumerate(params):
            state['wc'][i].copy_(w.data)
            lp['mw'][i].copy_(w.data)
            lp['mdw'][i].zero_()
            lp['eta'][i].normal_()

        state['debug'] = dict(wwpd=0, df=0, dF=0, g=0, eta=0)
        llr, beta1 = lp['llr'], lp['beta1']
        g = g0*(1+g1)**state['t']
        state['t'] += 1

        for i in range(L):
            f,_,_ = closure(prm, model, criterion, iterators, data_loaders, optimizer)
            for wc,w,mw,mdw,eta in zip(state['wc'], params,
                                    lp['mw'], lp['mdw'], lp['eta']):
                dw = w.grad.data

                if wd > 0:
                    dw.add_(wd, w.data)
                if mom > 0:
                    mdw.mul_(mom).add_(1-damp, dw)
                    if nesterov:
                        dw.add_(mom, mdw)
                    else:
                        dw = mdw

                # add noise
                eta.normal_()
                dw.add_(-g, wc-w.data).add_(eps/(np.sqrt(0.5*llr)), eta)

                # update weights
                w.data.add_(-llr, dw)
                mw.mul_(beta1).add_(1-beta1, w.data)

        if L > 0:
            # copy model back
            for i,w in enumerate(params):
                w.data.copy_(state['wc'][i])
                # w.grad.data.copy_(w.data-lp['mw'][i])
                w.grad.data.copy_(lp['mw'][i])


        for w,mdw,mw,eta in zip(params, state['mdw'], lp['mw'], lp['eta']):
            dw = w.grad.data
            if L > 0:
                # dw = (1+1e-2)*w.data - dw
                dw = deepcopy(w.data) - dw

            if wd > 0:
                dw.add_(wd, w.data)
            if mom > 0:
                mdw.mul_(mom).add_(1-damp, dw)
                if nesterov:
                    dw.add_(mom, mdw)
                else:
                    dw = mdw

            eta.normal_()
            w.data.add_(-lr, dw).add_(np.sqrt(0.5*lr)/100000, eta)
            # w.data.add_(-lr, dw)

        return mf
