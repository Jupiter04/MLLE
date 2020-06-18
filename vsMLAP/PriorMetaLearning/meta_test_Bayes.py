
from __future__ import absolute_import, division, print_function

import timeit

from Models.stochastic_models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import get_bayes_task_objective, run_test_Bayes
from Utils.common import grad_step, count_correct, get_loss_criterion, write_to_log
from PriorMetaLearning.Get_Objective_MPB import get_objective
from EntropySGD import object
from EntropySGD import optim
# import torch.optim as optim
import math


def run_learning(task_data, prior_model, prm, init_from_prior=True, verbose=1):

    # prm.optim_func, prm.optim_args = optim.EntropySGD, {'llr':0.01, 'lr':0.1, 'momentum':0.9, 'damp':0, 'weight_decay':1e-3, 'nesterov':True,
    #                  'L':20, 'eps':1e-3, 'g0':1e-4, 'g1':1e-3}

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------
    # Unpack parameters:
    # prm.optim_args['llr'] = 0.1
    # prm.optim_args['L'] = 20
    # # prm.optim_args['weight_decay'] = 1e-3
    # # prm.optim_args['g1'] = 0
    # prm.optim_args['g0'] = 1e-4
    optim_func, optim_args, lr_schedule =\
        prm.optim_func, prm.optim_args, prm.lr_schedule_test

    # prm.optim_func, prm.optim_args = optim.Adam, {'lr': prm.lr}  # 'weight_decay': 1e-4

    # lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [15, 20]}

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type)

    # Create posterior model for the new task:
    post_model = get_model(prm)

    if init_from_prior:
        post_model.load_state_dict(prior_model.state_dict())

        # prior_model_dict = prior_model.state_dict()
        # post_model_dict = post_model.state_dict()
        #
        # # filter out unnecessary keys:
        # prior_model_dict = {k: v for k, v in prior_model_dict.items() if '_log_var' in k or '_mu' in k}
        # # overwrite entries in the existing state dict:
        # post_model_dict.update(prior_model_dict)
        #
        # # #  load the new state dict
        # post_model.load_state_dict(post_model_dict)

        # add_noise_to_model(post_model, prm.kappa_factor)

    # The data-sets of the new task:
    train_loader = task_data
    test_loader = task_data['test']
    # n_train_samples = len(train_loader['train'].dataset)
    n_batches = len(train_loader)

    #  Get optimizer:
    optimizer = optim_func(filter(lambda p: p.requires_grad, post_model.parameters()), optim_args)
    # optimizer = optim_func(filter(lambda p: p.requires_grad, post_model.parameters()), optim_args['lr'])


    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------

    def run_train_epoch(i_epoch):
        # log_interval = 500

        post_model.train()

        train_iterators = iter(train_loader['train'])

        for batch_idx, batch_data in enumerate(train_loader['train']):

            task_loss, info = get_objective(prior_model, prm, [train_loader], object.feval,
                                                 [train_iterators], [post_model], loss_criterion, 1, [0])

            grad_step(task_loss[0], post_model, loss_criterion,
                      optimizer, prm, train_iterators, train_loader['train'], lr_schedule, prm.optim_args['lr'], i_epoch)

            # for log_var in post_model.parameters():
            #     if log_var.requires_grad is False:
            #         log_var.data = log_var.data - (i_epoch + 1) * math.log(1 + prm.gamma1)

            # Print status:
            log_interval = 10
            if (batch_idx) % log_interval == 0:
                batch_acc = info['correct_count'] / info['sample_count']
                print(cmn.status_string(i_epoch, prm.n_meta_train_epochs, batch_idx, n_batches, batch_acc) +
                      ' Empiric-Loss: {:.4f}'.format(info['avg_empirical_loss']))


    # -----------------------------------------------------------------------------------------------------------#
    # Update Log file
    if verbose == 1:
        write_to_log('Total number of steps: {}'.format(n_batches * prm.n_meta_test_epochs), prm)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    # Training loop:
    for i_epoch in range(prm.n_meta_test_epochs):
        run_train_epoch(i_epoch)

    # Test:
    test_acc, test_loss = run_test_Bayes(post_model, test_loader, loss_criterion, prm)

    stop_time = timeit.default_timer()
    cmn.write_final_result(test_acc, stop_time - start_time, prm, result_name=prm.test_type, verbose=verbose)

    test_err = 1 - test_acc
    return test_err, post_model
