


from __future__ import absolute_import, division, print_function

from datetime import datetime
import os
from torch.autograd import Variable
import torch.nn as nn
# from torch.nn.utils import nn
import torch
import numpy as np
import random
import sys
import pickle
from EntropySGD import object
from copy import deepcopy
import copy
# -----------------------------------------------------------------------------------------------------------#
# General - PyTorch
# -----------------------------------------------------------------------------------------------------------#


def get_value(x):
    ''' Returns the value of any scalar type'''
    if isinstance(x, Variable):
        if hasattr(x, 'item'):
            return x.item()
        else:
            return x.data[0]
    else:
        return x


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Get the parameters from a model:
def get_param_from_model(model, param_name):
    return [param for (name, param) in model.named_parameters() if name == param_name][0]

def zeros_gpu(size):
    if not isinstance(size, tuple):
        size = (size,)
    return torch.cuda.FloatTensor(*size).fill_(0)

def randn_gpu(size, mean=0, std=1):
    if not isinstance(size, tuple):
        size = (size,)
    return torch.cuda.FloatTensor(*size).normal_(mean, std)


def count_correct(outputs, targets):
    ''' Deterimne the class prediction by the max output and compare to ground truth'''
    pred = outputs.data.max(1, keepdim=True)[1] # get the index of the max output
    return get_value(pred.eq(targets.data.view_as(pred)).cpu().sum())


def correct_rate(outputs, targets):
    n_correct = count_correct(outputs, targets)
    n_samples = get_value(outputs.size()[0])
    return n_correct / n_samples


def save_model_state(model, f_path):

    with open(f_path, 'wb') as f_pointer:
        torch.save(model.state_dict(), f_pointer)
    return f_path


def load_model_state(model, f_path):
    if not os.path.exists(f_path):
        raise ValueError('No file found with the path: ' + f_path)
    with open(f_path, 'rb') as f_pointer:
        model.load_state_dict(torch.load(f_pointer))

#
# def get_data_path():
#     data_path_file = '../DataPath'
#     pth = '../data' #  default path
#     lines = open(data_path_file, "r").readlines()
#     if len(lines) > 1:
#         read_pth = lines[1]
#         father_dir = os.path.dirname(read_pth)
#         if father_dir is '~' or os.path.exists(father_dir):
#             pth = read_pth
#     print('Data path: ', pth)
#     return pth



# -------------------------------------------------------------------------------------------
#  Regularization
# -------------------------------------------------------------------------------------------
#
# def net_norm(model, p=2):
#     if p == 1:
#         loss_crit = torch.nn.L1Loss(size_average=False)
#     elif p == 2:
#         loss_crit = torch.nn.MSELoss(size_average=False)
#     else:
#         raise ValueError('Unsupported p')
#     total_norm = 0
#     for param in model.parameters():
#         target = Variable(zeros_gpu(param.size()), requires_grad=False)  # dummy target
#         total_norm += loss_crit(param, target)
#     return total_norm

def net_norm(model, p=2):
    total_norm = Variable(zeros_gpu(1), requires_grad=True)
    for param in model.parameters():
        total_norm = total_norm + param.pow(p).sum()
    return total_norm

# def clip_grad_by_norm_(grad, max_norm):
#     """
#     in-place gradient clipping.
#     :param grad: list of gradients
#     :param max_norm: maximum norm allowable
#     :return:
#     """
#
#     total_norm = 0
#     counter = 0
#     for g in grad:
#         param_norm = g.data.norm(2)
#         total_norm += param_norm.item() ** 2
#         counter += 1
#     total_norm = total_norm ** (1. / 2)
#
#     clip_coef = max_norm / (total_norm + 1e-6)
#     if clip_coef < 1:
#         for g in grad:
#             g.data.mul_(clip_coef)
#
#     return total_norm/counter


# -----------------------------------------------------------------------------------------------------------#
# Optimizer
# -----------------------------------------------------------------------------------------------------------#
# Gradient step function:
def grad_step(objective, model, criterion, optimizer, prm, iterators, data_loaders, lr_schedule=None, initial_lr=None, i_epoch=None):
    initial_lr = prm.optim_args['lr']
    if lr_schedule:
        adjust_learning_rate_schedule(optimizer, i_epoch, initial_lr, **lr_schedule)
    optimizer.zero_grad()
    objective.backward()
    # torch.nn.utils.clip_grad_norm(parameters, 0.25)
    optimizer.step(object.feval, prm, model, criterion, optimizer, iterators, data_loaders, i_epoch)
    # optimizer.step()

def prior_get_grad(prior_optimizer, posterior_optimizer):
    prior_params = prior_optimizer.param_groups[0]['params']
    posterior_params = posterior_optimizer.param_groups[0]['params']
    for w_prior, w_posterior in zip(prior_params, posterior_params):
        mu = deepcopy(w_posterior.grad.data)
        # dw = (w_prior.data - mu)
        dw = mu
        # dw = 100 * (w_prior.data - mu) / prm.batch_size
        w_prior.grad.data.add_(dw)

def prior_update(prior_optimizer,n_train_tasks,prm):
    prior_params = prior_optimizer.param_groups[0]['params']
    for w_prior in prior_params:
        mu = deepcopy(w_prior.grad.data)
        w_prior.data.zero_()
        w_prior.data.add_(1/(prm.kh), mu)
        a = n_train_tasks/prm.kh + 1/prm.ks
        w_prior.data.mul_(1/a)

def prior_updates(prior_optimizer, n_train_tasks, prm, lr_schedule=None, initial_lr=None, i_epoch=None):
    if lr_schedule:
        prior_lr = adjust_prior_learning_rate_schedule(prior_optimizer,i_epoch, initial_lr, **lr_schedule)
    else:
        prior = prm.prior_lr
    prior_params = prior_optimizer.param_groups[0]['params']
    # if not 'mdw' in prior_optimizer.state:
    #     prior_optimizer.state['mdw'] = []
    #     for w_prior in prior_params:
    #         prior_optimizer.state['mdw'].append(w_prior.grad.data)    
    # prior_params = prior_optimizer.param_groups[0]['params']
    for w_prior in prior_params:
        eta = deepcopy(w_prior.data)
        eta.normal_().detach()
        w_prior.data = w_prior.grad.data/(n_train_tasks+0.1)  #100
        # w_prior.data = w_prior.grad.data/(n_train_tasks+100) + np.sqrt(prm.prior_lr)/10000*eta

def prior_grad_step(prior_optimizer, n_train_tasks, prm, lr_schedule=None, initial_lr=None, i_epoch=None):
    if lr_schedule:
        prior_lr = adjust_prior_learning_rate_schedule(prior_optimizer, i_epoch, initial_lr, **lr_schedule)
    else:
        prior_lr = prm.prior_lr
    prior_params = prior_optimizer.param_groups[0]['params']
    for i,w_prior in enumerate(prior_params):
        # mu = deepcopy(w_prior.grad.data)
        ww = deepcopy(w_prior.data)
        # w_prior.grad.data.add_(w_prior.data)
        w_prior.grad.data.mul_(1/(n_train_tasks*prm.kh))
        w_prior.grad.data.add_(1/(n_train_tasks*prm.ks), ww)
        # print('{:}th, w_grad: {:.3}, w: {:.3}'.format(i,torch.mean(w_prior.grad.data), torch.mean(ww)))
    torch.nn.utils.clip_grad_norm(prior_params, 500)
    if not 'mdw' in prior_optimizer.state:
        prior_optimizer.state['mdw'] = []
        for w_prior in prior_params:
            prior_optimizer.state['mdw'].append(w_prior.grad.data)
    for w_prior, mdw in zip(prior_params, prior_optimizer.state['mdw']):
        # # nesterov + momentum
        # mdw.mul_(0.1).add_(w_prior.grad.data)
        # w_prior.grad.data.add_(0.1, mdw)
        eta = deepcopy(w_prior.data)
        eta.normal_()
        w_prior.data.add_(-prior_lr, w_prior.grad.data).add_(np.sqrt(prm.prior_lr) / 10000, eta)
        # w_prior.data.add_(-prior_lr, w_prior.grad.data)
    # for posterior_optimizer in posterior_optimizers:
    #     posterior_params = posterior_optimizer.param_groups[0]['params']
    #     for w_prior, w_posterior, mdw in zip(prior_params, posterior_params, prior_optimizer.state['mdw']):
    #         mu = deepcopy(w_posterior.grad.data)
    #         eta = deepcopy(w_posterior.grad.data)
    #         eta.normal_()
    #         dw = (w_prior.data - mu)/prm.kh
    #         ww = deepcopy(w_prior.data)
    #         dw = -(dw + ww/(prm.ks*n_train_tasks))/(2*n_train_tasks)
    #         # dw = deepcopy(dd)
    #         # # nesterov + momentum
    #         # mdw.mul_(0.9).add_(dw)
    #         # dw.add_(0.9, mdw)
    #         w_prior.grad.data.add_(dw)
            # w_prior.grad.data.add_(1 / (2 * n_train_tasks * prm.ks), -ww)
    # for w_prior in prior_params:
    #     eta = deepcopy(w_prior.data)
    #     eta.normal_()
    #     # ww = deepcopy(w_prior.data)
    #     # w_prior.data.add_(np.sqrt(prior_lr) / 100, eta).add_(prior_lr/(2*n_train_tasks*prm.ks), -ww)
    #     # w_prior.grad.data.add_(1 / (2 * n_train_tasks * prm.ks), -ww)
    #     w_prior.data.add_(prior_lr, w_prior.grad.data).add_(np.sqrt(prior_lr) / 10000, eta)
        # w_prior.data.add_(prior_lr, w_prior.grad.data)
        # w_prior.data.add_(np.sqrt(prior_lr) / 100, eta)

def adjust_learning_rate_interval(optimizer, epoch, initial_lr, gamma, decay_interval):
    """Sets the learning rate to the initial LR decayed by gamma every decay_interval epochs"""
    lr = initial_lr * (gamma ** (epoch // decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_schedule(optimizer, epoch, initial_lr, decay_factor, decay_epochs):
    """The learning rate is decayed by decay_factor at each interval start """

    # Find the index of the current interval:
    interval_index = len([mark for mark in decay_epochs if mark < epoch])

    lr = initial_lr * (decay_factor ** interval_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_prior_learning_rate_schedule(optimizer, epoch, initial_lr, decay_factor, decay_epochs):
    """The learning rate is decayed by decay_factor at each interval start """

    # Find the index of the current interval:
    interval_index = len([mark for mark in decay_epochs if mark < epoch])

    lr = initial_lr * (decay_factor ** interval_index)
    return lr
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr

# -----------------------------------------------------------------------------------------------------------#
#  Configuration
# -----------------------------------------------------------------------------------------------------------#

def get_loss_criterion(loss_type):
# Note: the loss use the un-normalized net outputs (scores, not probabilities)

    criterion_dict = {'CrossEntropy':nn.CrossEntropyLoss(size_average=True).cuda(),
                 'L2_SVM':nn.MultiMarginLoss(p=2, margin=1, weight=None, size_average=True)}

    return criterion_dict[loss_type]


# -----------------------------------------------------------------------------------------------------------#
# Prints
# -----------------------------------------------------------------------------------------------------------#

def status_string(i_epoch, num_epochs, batch_idx, n_batches, batch_acc):

    progress_per = 100. * (i_epoch * n_batches + batch_idx) / (n_batches * num_epochs)
    # return ('({:2.1f}%)\tEpoch: {:3} \t Batch: {:4} \t Objective: {:.4} \t  Acc: {:1.3}\t'.format(
    #     progress_per, i_epoch, batch_idx, loss_data, batch_acc))
    return ('({:2.1f}%)\tEpoch: {} \t Batch: {} \t  Acc: {:1.3f}\t'.format(
        progress_per, i_epoch, batch_idx, float(batch_acc)))

# def status_string_meta(i_epoch, prm, batch_acc, loss_data):
#
#     progress_per = 100. * (i_epoch ) / ( prm.num_epochs)
#     return ('({:2.1f}%)\tEpoch: {:3} \t Objective: {:.4} \t  Acc: {:1.3}\t'.format(
#         progress_per, i_epoch + 1, loss_data, batch_acc))


def get_model_string(model):
    return str(model.model_type) + '-' + str(model.model_name)  # + ':' + '-> '.join([m.__str__() for m in model._modules.values()])

# -----------------------------------------------------------------------------------------------------------#
# Result saving
# -----------------------------------------------------------------------------------------------------------#

def create_result_dir(prm):
    # If run_name empty, set according to time
    time_str = datetime.now().strftime(' %Y-%m-%d %H:%M:%S')
    if prm.run_name == '':
        prm.run_name = 'log' + time_str
    prm.result_dir = os.path.join('saved', prm.run_name)
    if not os.path.exists(prm.result_dir):
        os.makedirs(prm.result_dir)
    message = ['Log file created at ' + time_str,
               'Run script: ' + sys.argv[0],
               'Parameters:', str(prm), '-'*50]
    write_to_log(message, prm, mode='w') # create new log file
    # set the path to pre-trained model, in case it is loaded (if empty - set according to run_name)
    if not hasattr(prm, 'load_model_path') or prm.load_model_path == '':
        prm.load_model_path = os.path.join(prm.result_dir, 'model.pt')


def write_to_log(message, prm, mode='a', update_file=True):
    # mode='a' is append
    # mode = 'w' is write new file
    if not isinstance(message, list):
        message = [message]
    # update log file:
    if update_file:
        log_file_path = os.path.join(prm.result_dir, 'log') + '.out'
        with open(log_file_path, mode) as f:
            for string in message:
                print(string, file=f)
    # print to console:
    for string in message:
        print(string)

def write_final_result(test_acc, run_time, prm, result_name='', verbose=1):
    message = []
    if verbose == 1:
        message.append('Run finished at: ' + datetime.now().strftime(' %Y-%m-%d %H:%M:%S'))
    message.append(result_name + ' Average Test Error: {:.3}%\t Runtime: {} [sec]'
                     .format(100 * (1 - test_acc), run_time))
    write_to_log(message, prm)


def save_run_data(prm, info_dict):
    run_data_file_path = os.path.join(prm.result_dir, 'run_data.pkl')
    with open(run_data_file_path, 'wb') as f:
        pickle.dump([prm, info_dict], f)


def load_run_data(result_dir):
    run_data_file_path = os.path.join(result_dir, 'run_data.pkl')
    with open(run_data_file_path, 'rb') as f:
       prm, info_dict = pickle.load(f, encoding='iso-8859-1')
    return prm, info_dict


# def save_code(setting_name, run_name):
#     dir_name = setting_name + '_' + run_name
#     # Create backup of code
#     source_dir = os.getcwd()
#     dest_dir = source_dir + '/Code_Archive/' + dir_name # os.path.join
#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)
#     for filename in glob.glob(os.path.join(source_dir, '*.*')):
#         shutil.copy(filename, dest_dir)

def transfer_weights(model_from, model_to):
    wf = copy.deepcopy(model_from.state_dict())
    wt = model_to.state_dict()
    for k in wt.keys() :
        if (not k in wf):
            wf[k] = wt[k]
    model_to.load_state_dict(wf)
    return model_to

