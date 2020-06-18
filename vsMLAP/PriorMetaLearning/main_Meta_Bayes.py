
from __future__ import absolute_import, division, print_function
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import argparse
import timeit, time, os
import numpy as np
import torch
# import torch.optim as optim

from PriorMetaLearning import meta_test_Bayes, meta_train_Bayes_finite_tasks, meta_train_Bayes_infinite_tasks
from Data_Path import get_data_path
from Models.stochastic_models import get_model
from Utils.data_gen import Task_Generator
from Utils.common import save_model_state, load_model_state, create_result_dir, set_random_seed, write_to_log, save_run_data
from PriorMetaLearning.Analyze_Prior import run_prior_analysis

from EntropySGD import optim
# import torch.optim as optim
torch.backends.cudnn.benchmark = True  # For speed improvement with models with fixed-length inputs
# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

# ----- Run Parameters ---------------------------------------------#

parser.add_argument('--run-name', type=str, help='Name of dir to save results in (if empty, name by time)',
                    default='')

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--mode', type=str, help='MetaTrain or LoadMetaModel',
                    default='MetaTrain')   # 'MetaTrain'  \ 'LoadMetaModel'

parser.add_argument('--load_model_path', type=str, help='set the path to pre-trained model, in case it is loaded (if empty - set according to run_name)',
                    default='')

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing (reduce if memory is limited)',
                    default=128) #mnist:128 cifar10:32

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)#mnist:128 cifar10:32

parser.add_argument('--n_test_tasks', type=int,
                    help='Number of meta-test tasks for meta-evaluation (how many tasks to average in final result)',
                    default=20)

# ----- Task Parameters ---------------------------------------------#

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'CIFAR10' / Omniglot / SmallImageNet",
                    default='MNIST')

parser.add_argument('--n_train_tasks', type=int, help='Number of meta-training tasks (0 = infinite)',
                    default=0)

parser.add_argument('--data-transform', type=str, help="Data transformation:  'None' / 'Permute_Pixels' / 'Permute_Labels'/ Shuffled_Pixels ",
                    default='Permute_Labels')

parser.add_argument('--n_pixels_shuffles', type=int, help='In case of "Shuffled_Pixels": how many pixels swaps',
                    default=200)

parser.add_argument('--limit_train_samples_in_test_tasks', type=int,
                    help='Upper limit for the number of training samples in the meta-test tasks (0 = unlimited)',
                    default=2000)

# N-Way K-Shot Parameters:
parser.add_argument('--N_Way', type=int, help='Number of classes in a task (for Omniglot)',
                    default=5)

parser.add_argument('--K_Shot_MetaTrain', type=int,
                    help='Number of training sample per class in meta-training in N-Way K-Shot data sets',
                    default=1)  # Note:  test samples are the rest of the data

parser.add_argument('--K_Shot_MetaTest', type=int,
                    help='Number of training sample per class in meta-testing in N-Way K-Shot data sets',
                    default=1)  # Note:  test samples are the rest of the data

# SmallImageNet Parameters:
parser.add_argument('--n_meta_train_classes', type=int,
                    help='For SmallImageNet: how many categories are available for meta-training',
                    default=500)

# Omniglot Parameters:
parser.add_argument('--chars_split_type', type=str,
                    help='how to split the Omniglot characters  - "random" / "predefined_split"',
                    default='random')

parser.add_argument('--n_meta_train_chars'
                    , type=int, help='For Omniglot: how many characters to use for meta-training, if split type is random',
                    default=1200)

# ----- Algorithm Parameters ---------------------------------------------#

parser.add_argument('--complexity_type', type=str,
                    help=" The learning objective complexity type",
                    default='NewBoundSeeger')  #  'NoComplexity' /  'Variational_Bayes' / 'PAC_Bayes_Pentina'   NewBoundMcAllaster / NewBoundSeeger'"

# parser.add_argument('--override_eps_std', type=float,
#                     help='For debug: set the STD of epsilon variable for re-parametrization trick (default=1.0)',
#                     default=1.0)

parser.add_argument('--loss-type', type=str, help="Loss function",
                    default='CrossEntropy') #  'CrossEntropy' / 'L2_SVM'

parser.add_argument('--model-name', type=str, help="Define model type (hypothesis class)'",
                    default='AllCnn')  # OmConvNet / 'FcNet3' / 'ConvNet3' /'AllCnn'

parser.add_argument('--n_meta_train_epochs', type=int, help='number of epochs to train',
                    default=5)

parser.add_argument('--n_inner_steps', type=int,
                    help='For infinite tasks case, number of steps for training per meta-batch of tasks',
                    default=1)  #

parser.add_argument('--n_meta_test_epochs', type=int, help='number of epochs to train',
                    default=10)  #

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)

# parser.add_argument('--log_gamma0', type=float, help='log_var_init',
#                     default=-9)

# parser.add_argument('--gamma1', type=float, help='gamma1',
#                     default=1e-2)

parser.add_argument('--meta_batch_size', type=int, help='Maximal number of tasks in each meta-batch',
                    default=32)
# -------------------------------------------------------------------------------------------

prm = parser.parse_args()
prm.data_path = get_data_path()
set_random_seed(prm.seed)
create_result_dir(prm)


# Weights initialization (for Bayesian net):
# mnist:-8
prm.log_var_init = {'mean': -8, 'std': 0.1} # The initial value for the log-var parameter (rho) of each weight

# Number of Monte-Carlo iterations (for re-parametrization trick):
prm.n_MC = 1

#  Define optimizer:
# mnist
prm.optim_func, prm.optim_args = optim.EntropySGD, {'llr':0.2, 'lr':1, 'momentum':0, 'damp':0, 'weight_decay':0, 'nesterov':True,
                 'L':20, 'eps':1e-3, 'g0':1e-4, 'g1':0.001}
# # cifar10
# prm.optim_func, prm.optim_args = optim.EntropySGD, {'llr':0.1, 'lr':1, 'momentum':0, 'damp':0, 'weight_decay':0, 'nesterov':True,
#                  'L':20, 'eps':1e-4, 'g0':0.03, 'g1':1e-3}

# prm.optim_func, prm.optim_args = optim.Adam,  {'lr': 1e-3}

print(prm.log_var_init['mean'], prm.optim_args['g0'])

prm.prior_lr = 0.01

prm.kh = 0.001 # square  0.001
prm.ks = 500   # square   500
# prm.optim_func, prm.optim_args  = optim.Adam,  {'lr': prm.lr} #'weight_decay': 1e-4
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule for mnist:
prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [1,2,3,4,5]}  # [1,2,3,4,5]
# prm.lr_schedule = {}
prm.lr_schedule_test = {'decay_factor': 0.01, 'decay_epochs': [1,2,3,4,5]} # [2,4,6,8,10]
# prm.lr_schedule_test = {}  # No decay
# prm.prior_lr_schedule = {'decay_factor': 0.5, 'decay_epochs': [1,2,3,4,5]} # [5,10,15,20]  [10,20,30,40,50]
prm.prior_lr_schedule = {}

# # Learning rate decay schedule for cifar10:
# prm.lr_schedule = {'decay_factor': 0.2, 'decay_epochs': [4,8,12]}  # [1,2,3,4,5]
# # prm.lr_schedule = {}
# prm.lr_schedule_test = {'decay_factor': 0.2, 'decay_epochs': [4,8,12]} # [2,4,6,8,10]
# # prm.lr_schedule_test = {}  # No decay
# prm.prior_lr_schedule = {'decay_factor': 0.5, 'decay_epochs': [4,8,12]} # [5,10,15,20]  [10,20,30,40,50]
# # prm.prior_lr_schedule = {}

init_from_prior = True  #  False \ True . In meta-testing -  init posterior from learned prior

# Test type:
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote' / 'AvgVote'

# path to save the learned meta-parameters
save_path = os.path.join(prm.result_dir, 'model.pt')

task_generator = Task_Generator(prm)

# -------------------------------------------------------------------------------------------
#  Run Meta-Training
# -------------------------------------------------------------------------------------------

start_time = timeit.default_timer()

# prm.mode = 'LoadMetaModel'

if prm.mode == 'MetaTrain':

    n_train_tasks = prm.n_train_tasks
    if n_train_tasks:
        # In this case we generate a finite set of train (observed) task before meta-training.
        # Generate the data sets of the training tasks:
        write_to_log('--- Generating {} training-tasks'.format(n_train_tasks), prm)
        train_data_loaders = task_generator.create_meta_batch(prm, n_train_tasks, meta_split='meta_train')

        # Meta-training to learn prior:
        prior_model = meta_train_Bayes_finite_tasks.run_meta_learning(train_data_loaders, prm)
        # save learned prior:
        save_model_state(prior_model, save_path)
        write_to_log('Trained prior saved in ' + save_path, prm)
    else:
        # In this case we observe new tasks generated from the task-distribution in each meta-iteration.
        write_to_log('---- Infinite train tasks - New training tasks are '
                     'drawn from tasks distribution in each iteration...', prm)

        # Meta-training to learn meta-prior (theta params):
        prior_model = meta_train_Bayes_infinite_tasks.run_meta_learning(task_generator, prm)


elif prm.mode == 'LoadMetaModel':

    # Loads  previously training prior.
    # First, create the model:
    prior_model = get_model(prm)
    prm.load_model_path = '/hdd/shiwei/meta_learning _example/PriorMetaLearning/saved/ShuffledPixels100_TasksN/log 2019-05-23 14:36:30/1/model.pt'
    # prm.load_model_path = '/hdd/shiwei/meta_learning _example/PriorMetaLearning/saved/PermutedLabels_TasksN/log 2019-04-18 12:40:53/5/model.pt'
    # prm.load_model_path = '/hdd/shiwei/meta_learning _example/PriorMetaLearning/saved/model.pt'
    # Then load the weights:
    load_model_state(prior_model, prm.load_model_path)
    write_to_log('Pre-trained  prior loaded from ' + prm.load_model_path, prm)
else:
    raise ValueError('Invalid mode')

# -------------------------------------------------------------------------------------------
# Generate the data sets of the test tasks:
# -------------------------------------------------------------------------------------------

n_test_tasks = prm.n_test_tasks

limit_train_samples_in_test_tasks = prm.limit_train_samples_in_test_tasks
if limit_train_samples_in_test_tasks == 0:
    limit_train_samples_in_test_tasks = None

write_to_log('---- Generating {} test-tasks with at most {} training samples'.
             format(n_test_tasks, limit_train_samples_in_test_tasks), prm)

# test_tasks_data  = task_generator.create_meta_batch(prm, n_test_tasks, meta_split='meta_test')
test_tasks_data = task_generator.create_meta_batch(prm, n_test_tasks, meta_split='meta_test',
                                                   limit_train_samples=limit_train_samples_in_test_tasks)
# #
# -------------------------------------------------------------------------------------------
#  Run Meta-Testing
# -------------------------------------------------------------------------------
write_to_log('Meta-Testing with transferred prior....', prm)

test_err_vec = np.zeros(n_test_tasks)
for i_task in range(n_test_tasks):
    print('Meta-Testing task {} out of {}...'.format(1+i_task, n_test_tasks))
    task_data = test_tasks_data[i_task]
    test_err_vec[i_task], _ = meta_test_Bayes.run_learning(task_data, prior_model, prm, init_from_prior, verbose=0)


# save result
save_run_data(prm, {'test_err_vec': test_err_vec})

# -------------------------------------------------------------------------------------------
#  Print results
# -------------------------------------------------------------------------------------------
#  Print prior analysis
run_prior_analysis(prior_model)

stop_time = timeit.default_timer()
write_to_log('Total runtime: ' +
             time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)),  prm)

#  Print results
write_to_log('----- Final Results: ', prm)
write_to_log('----- Meta-Testing - Avg test err: {:.3}%, STD: {:.3}%'
             .format(100 * test_err_vec.mean(), 100 * test_err_vec.std()), prm)

# -------------------------------------------------------------------------------------------
#  Compare to standard learning
# -------------------------------------------------------------------------------------------
# from Single_Task import learn_single_standard
# test_err_standard = np.zeros(n_test_tasks)
# for i_task in range(n_test_tasks):
#     print('Standard learning task {} out of {}...'.format(i_task, n_test_tasks))
#     task_data = test_tasks_data[i_task]
#     test_err_standard[i_task], _ = learn_single_standard.run_learning(task_data, prm, verbose=0)
#
# write_to_log('Standard - Avg test err: {:.3}%, STD: {:.3}%'.
#              format(100 * test_err_standard.mean(), 100 * test_err_standard.std()), prm)
