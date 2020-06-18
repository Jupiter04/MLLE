from __future__ import absolute_import, division, print_function
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import numpy as np
from subprocess import call
import argparse
import pickle, os, timeit, time
import matplotlib.pyplot as plt
from datetime import datetime
from Utils.common import load_run_data



base_run_name = 'PermutedLabels_TasksN'
# base_run_name = 'ShuffledPixels_TaskN'

min_n_tasks = 5  # 1
max_n_tasks = 5  # 10

run_experiments = True # If false, just analyze the previously saved experiments

root_saved_dir = 'saved/'
# -------------------------------------------------------------------------------------------
# Run experiments for a grid of 'number of training tasks'
# -------------------------------------------------------------------------------------------

n_tasks_vec = np.arange(min_n_tasks, max_n_tasks+1)

time_str = datetime.now().strftime(' %Y-%m-%d %H:%M:%S')

sub_runs_names = [base_run_name + '/' + 'log' + time_str + '/' + str(n_train_tasks) for n_train_tasks in n_tasks_vec]


if run_experiments:
    start_time = timeit.default_timer()
    for i_run, n_train_tasks in enumerate(n_tasks_vec):
        call(['python', 'main_Meta_Bayes.py',
              '--run-name', sub_runs_names[i_run],
              '--data-source', 'MNIST',  # 'Omniglot'/ 'MNIST' / 'CIFAR10'
              '--data-transform', 'Permute_Labels',  # 'Shuffled_Pixels' 'Permute_Labels', 'None'
              '--n_train_tasks', str(n_train_tasks),
              '--limit_train_samples_in_test_tasks', '2000',
              '--model-name',   'ConvNet3',  # 'FcNet3' 'ConvNet3', 'OmConvNet', 'AllCnn'
              # '--complexity_type',  'NewBoundSeeger',
              '--n_test_tasks', '20',  # 20 100
              '--n_meta_train_epochs', '5',  # mnist:5, cifar10:10
              '--n_meta_test_epochs', '5',  # mnist:5, cifar10:10
              '--meta_batch_size', '1',  # mnist:16
              '--mode', 'MetaTrain',  # 'MetaTrain'  \ 'LoadMetaModel'
              ])
    stop_time = timeit.default_timer()
    # Save log text
    message = ['Run finished at ' +  datetime.now().strftime(' %Y-%m-%d %H:%M:%S'),
               'Tasks number grid: ' + str(n_tasks_vec),
                'Total runtime: ' + time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)),
                '-'*30]
    log_time_str = 'log' + time_str
    log_file_path = os.path.join(root_saved_dir, base_run_name, log_time_str, 'log') + '.out'
    with open(log_file_path, 'a') as f:
        for string in message:
            print(string, file=f)
            print(string)

# -------------------------------------------------------------------------------------------
# Analyze the experiments
# -------------------------------------------------------------------------------------------

mean_error_per_tasks_n = np.zeros(len(n_tasks_vec))
std_error_per_tasks_n = np.zeros(len(n_tasks_vec))

for i_run, n_train_tasks in enumerate(n_tasks_vec):
    run_result_path = os.path.join(root_saved_dir, sub_runs_names[i_run])
    prm, info_dict = load_run_data(run_result_path)
    test_err_vec = info_dict['test_err_vec']
    mean_error_per_tasks_n[i_run] = test_err_vec.mean()
    std_error_per_tasks_n[i_run] = test_err_vec.std()

# Saving the analysis:
with open(os.path.join(root_saved_dir, base_run_name, log_time_str, 'runs_analysis.pkl'), 'wb') as f:
    pickle.dump([mean_error_per_tasks_n, std_error_per_tasks_n, n_tasks_vec], f)

# Plot the analysis:
# plt.figure()
# plt.errorbar(n_tasks_vec, 100*mean_error_per_tasks_n, yerr=100*std_error_per_tasks_n)
# plt.xticks(n_tasks_vec)
# plt.xlabel('Number of training-tasks')
# plt.ylabel('Error on new task [%]')
# plt.show()

