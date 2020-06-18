from __future__ import absolute_import, division, print_function


# from Models.stochastic_models import get_model
from Models.stochastic_models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import get_bayes_task_objective, run_test_Bayes, get_meta_complexity_term
from Utils.common import grad_step, net_norm, count_correct, get_loss_criterion, get_value

# -------------------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------------------
def get_objective(prior_model, prm, mb_data_loaders, feval, mb_iterators, mb_posteriors_models, loss_criterion, n_train_tasks, task_ids_in_meta_batch):
    '''  Calculate objective based on tasks in meta-batch '''
    # note: it is OK if some tasks appear several times in the meta-batch

    n_tasks_in_mb = len(mb_data_loaders)

    sum_empirical_loss = 0
    correct_count = 0
    sample_count = 0
    task_loss_list = []
    for i in range(n_train_tasks):
        task_loss_list.append(0)
    # all_task_loss = 0

    # ----------- loop over tasks in meta-batch -----------------------------------#
    for i_task in range(n_tasks_in_mb):

        # n_samples = mb_data_loaders[i_task]['n_train_samples']

        # The posterior model corresponding to the task in the batch:
        post_model = mb_posteriors_models[i_task]
        post_model.train()

        loss, cor_count, sum_count = feval(prm, post_model, loss_criterion, mb_iterators[i_task], mb_data_loaders[i_task]['train'])

        correct_count += cor_count
        sample_count += sum_count

        # Intra-task complexity of current task:
        task_loss_list[task_ids_in_meta_batch[i_task]] += loss
        # task_empirical_loss += loss

        sum_empirical_loss += loss

    # end loop over tasks in meta-batch
    avg_empirical_loss = (1 / n_tasks_in_mb) * sum_empirical_loss

    info = {'sample_count': get_value(sample_count), 'correct_count': get_value(correct_count),
                  'avg_empirical_loss': get_value(avg_empirical_loss)}
    return task_loss_list, info
