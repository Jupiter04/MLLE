import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
# from    torch import optim
import  numpy as np
from EntropySGD import optim

from    learner import Learner
from    copy import deepcopy



class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.meta_states = dict()
        self.iteration = 1

        self.net = Learner(config, args.imgc, args.imgsz)
        # self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.optimizer = optim.EntropySGD(self.net.parameters(),
            config=dict(lr=1.2, momentum=0, nesterov=True, weight_decay=0,
                L=10, eps=1e-4, g0=1e-2, g1=1e-1))
    
    def criterion(ypred,y):
        return F.cross_entropy(ypred, y)

    def train_on_batch_entropy_sgd(x,y):
        def helper(x,y):
            def feval(x,y):
                x = to_torch(x)
                y = to_torch(y)
                self.net.zero_grad()
                ypred = self.net(x)
                loss = criterion(ypred,y)
                loss.backward()
                # print(loss.data)
                # print('================')
                return loss.data
            return feval(x,y)
        f = self.optimizer.step(x, y, helper, self.net, criterion)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def update_meta_states():
        mu_after = dict()
        for name,params in self.net.named_parameters():
            mu_after[name] = params.grad
            # print(params.grad)
        # mu_after = model.named_parameters()
        for name,params in self.net.named_parameters():
            # print(params.grad)
            break
        # outerstepsize = outer_stepsize_reptile * 1
        if self.meta_states:
            self.net.load_state_dict({name: mu_after[name] for name in self.meta_states})

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        if self.meta_states:
            self.net.load_state_dict(self.meta_states)
	#else:
	#    self.meta_states = deepcopy(self.net.state_dict())
	
        mu_after = dict()    
        for i in range(task_num):
            self.net.load_state_dict(self.meta_states)
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            train_on_batch_entropy_sgd(x_spt[i], y_spt[i])
            # loss = F.cross_entropy(logits, y_spt[i])
            # grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = self.net.parameters()
            fast_states = deepcopy(self.net.state_dict())

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], meta_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct
	    
	    # mu_after = dict()
            # for name,params in self.net.named_parameters():
            #     mu_after[name] = params.grad
           
	    
            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct
	    
	    # weights_state = deepcopy(self.net.state_dict())

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                # logits = self.net(x_spt[i], fast_weights, bn_training=True)
                # loss = F.cross_entropy(logits, y_spt[i])
                train_on_batch_entropy_sgd(x_spt[i], y_spt[i])
                # 2. compute grad on theta_pi
                # grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                fast_weights = self.net.parameters()
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

                if mu_after:
                    for name,params in self.net.named_parameters():
                        mu_after[name] += (params.grad/task_num)
                else:
                    for name,params in model.named_parameters():
                        mu_after[name] = (params.grad/task_num)

        if self.iteration==0:
            self.meta_states = ({name: mu_after[name] for name in self.meta_states})
        else:
            self.meta_states = ({name: self.meta_states[name]*(1-1/(self.iteration+1)) + mu_after[name]/(1+self.iteration)
                                    for name in self.meta_states})
        self.iteration += 1
# end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        #self.meta_optim.zero_grad()
        #loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        #self.meta_optim.step()


        accs = np.array(corrects) / (querysz * task_num)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = self.net
        net.load_state_dict(self.meta_states)

        # 1. run the i-th task and compute loss for k=0
        #logits = net(x_spt)
        #loss = F.cross_entropy(logits, y_spt)
        #grad = torch.autograd.grad(loss, net.parameters())
        #fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
	    #train_on_batch_entropy_sgd(x_spt[i], y_spt[i])
                # 2. compute grad on theta_pi
                # grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
	    #fast_weights = self.net.parameters()
                
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            #logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            train_on_batch_entrooy_sgd(x_spt,y_spt)
            logits_q = net(x_qry, net.parameters(),bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            train_on_batch_entropy_sgd(x_qry,y_qry)
            logits = net(x_spt, net.parameters(), bn_training=True)
            # loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            #grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            #fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz

        return accs




def main():
    pass


if __name__ == '__main__':
    main()
