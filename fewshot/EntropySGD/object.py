
from torch.autograd import Variable
import torch

def get_batch_vars(batch_data, args, is_test=False):
    ''' Transform batch to variables '''
    inputs, targets = batch_data
    inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs, volatile=is_test), Variable(targets, volatile=is_test)
    return inputs, targets

def get_next_batch_cyclic(data_iterator, data_generator):
    ''' get sample from iterator, if it finishes then restart  '''
    try:
        batch_data = data_iterator.next()
    except StopIteration:
        # in case some task has less samples - just restart the iterator and re-use the samples
        data_iterator = iter(data_generator)
        batch_data = data_iterator.next()
    return batch_data

def get_value(x):
    ''' Returns the value of any scalar type'''
    if isinstance(x, Variable):
        if hasattr(x, 'item'):
            return x.item()
        else:
            return x.data[0]
    else:
        return x

def count_correct(outputs, targets):
    ''' Deterimne the class prediction by the max output and compare to ground truth'''
    pred = outputs.data.max(1, keepdim=True)[1] # get the index of the max output
    return get_value(pred.eq(targets.data.view_as(pred)).cpu().sum())

def grad_init(prm, model, loss_criterion, iterators, data_loaders, optimizer):
    batch_data = get_next_batch_cyclic(iterators, data_loaders)

    inputs,targets = get_batch_vars(batch_data, prm)

    outputs = model(inputs)
    optimizer.zero_grad()
    objective=loss_criterion(outputs, targets)
    objective.backward()
    optimizer.zero_grad()

def feval(prm, model, loss_criterion, iterators, data_loaders, optimizer=None):

    batch_data = get_next_batch_cyclic(iterators, data_loaders)

    inputs, targets = get_batch_vars(batch_data, prm)

    count = 0
    sum_count = 0
    f = 0.0
    for i_MC in range(prm.n_MC):
        outputs = model(inputs)
        f_cur = loss_criterion(outputs, targets)
        count += count_correct(outputs, targets)
        sum_count += inputs.size(0)
        f += f_cur * (1/prm.n_MC)

    if optimizer is not None:
        optimizer.zero_grad()
    if optimizer is not None:
        f.backward()

    return f, count, sum_count