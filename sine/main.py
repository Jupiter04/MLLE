import argparse
import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy
from EntropySGD import optim
from Models.stochastic_models import get_model
import argparse
from Models.common import net_norm
from Models.Bayes_utils import get_bayes_task_objective, get_meta_complexity_term

def experiment(run, plot=True):
    seed = 0
    inner_step_size = 0.02  # stepsize in inner SGD
    inner_step_size_mlap = 0.01  #0.01
    inner_epochs = 1  # number of epochs of each inner SGD
    outer_stepsize_reptile = 0.1  # stepsize of outer optimization, i.e., meta-optimization
    outer_stepsize_maml = 0.01
    outer_stepsize_mlap = 0.1  #0.01
    n_iterations = 300 # number of outer updates; each iteration we sample one task and update on it
    # entropy-sgd: 30000/10
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    result = np.zeros((4, n_iterations//10+1))

    # Define task distribution
    x_all = np.linspace(-5, 5, 50)[:, None]  # All of the x points
    n_train = 10  # Size of training minibatches

    def gen_task():
        "Generate classification problem"
        phase = rng.uniform(low=0, high=2 * np.pi)
        ampl = rng.uniform(0.1, 5)
        f_randomsine = lambda x: np.sin(x + phase) * ampl
        return f_randomsine

    # Define model. Reptile paper uses ReLU, but Tanh gives slightly better results
    model = nn.Sequential(
        nn.Linear(1, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )
#    sketch_model = nn.Sequential(
#        nn.Linear(1,64),
#        nn.Tanh(),
#        nn.Linear(64,64)
#        nn.Tanh(),
#        nn.Linear(64,1),
#        )
    prm=argparse.ArgumentParser()
    prm.model_name = 'FcNet3'
    prm.log_var_init = {'mean':-5, 'std':0.1}
    prm.complexity_type = 'NewBoundSeeger'
    prm.kappa_post = 0.1 # 5e-4
    prm.kappa_prior = 1  # 10
    
    prm.delta = 0.1
    sto_model = model
    # sto_model = get_model(prm)

    optimizer = optim.EntropySGD(model.parameters(),
            config=dict(lr=1.2, momentum=0, nesterov=True, weight_decay=0,
                L=10, eps=1e-4, g0=1e-2, g1=1e-1))

    def to_torch(x):
        return ag.Variable(torch.Tensor(x))
    
    def criterion(ypred, y):
        return (ypred-y).pow(2).mean()

    def train_on_batch_entropy_sgd(x,y):
        def helper(x,y):
            def feval(x,y):
                x = to_torch(x)
                y = to_torch(y)
                model.zero_grad()
                ypred = model(x)
                loss = (ypred-y).pow(2).mean()
                loss.backward()
                # print(loss.data)
                # print('================')
                return loss.data
            return feval(x,y)
        f = optimizer.step(x, y, helper, model, criterion)
        # print(f)
    
    def train_on_batch_mlap(x,y,post_model,train=True):
        x = to_torch(x)
        y = to_torch(y)
        post_model.zero_grad()
        sto_model.zero_grad()
        ypred = post_model(x)
        loss = (ypred-y).pow(2).mean()
        # print(loss)
        loss, task_complexity = get_bayes_task_objective(prm, sto_model, post_model, n_train, loss, 0, 1, True)
        hyper_kl = (1/(2*prm.kappa_prior**2))*net_norm(sto_model,p=2)
        meta_complex_term = get_meta_complexity_term(hyper_kl, prm, 1)
        loss = loss + task_complexity/n_train + meta_complex_term
        loss.backward()
        for param in post_model.parameters():
            param.data -= inner_step_size_mlap * param.grad.data
            # print(param.grad.data)
        if train:
            for param in sto_model.parameters():
                param.data -= outer_stepsize_mlap * param.grad.data
                # print(param.grad.data)

    def train_on_batch(x, y):
        x = to_torch(x)
        y = to_torch(y)
        model.zero_grad()
        ypred = model(x)
        loss = (ypred - y).pow(2).mean()
        loss.backward()
        for param in model.parameters():
            param.data -= inner_step_size * param.grad.data

    def predict(x, post_model=None):
        x = to_torch(x)
        if post_model:
            return post_model(x).data.numpy()
        else:
            return model(x).data.numpy()

    # Choose a fixed task and minibatch for visualization
    f_plot = gen_task()
    n_test = 1
    f_plot_list = [f_plot]
    #f_plot_list = [gen_task() for i in range(n_test)]
    xtrain_plot = x_all[rng.choice(len(x_all), size=n_train)]

    # Training loop
    for iteration in range(n_iterations):
        weights_before = deepcopy(model.state_dict())

        # Generate task
        f = gen_task()
        y_all = f(x_all)

        # Do SGD on this task
        inds = rng.permutation(len(x_all))
        train_ind = inds[:-1 * n_train]
        val_ind = inds[-1 * n_train:]       # Val contains 1/5th of the sine wave
        if run=='MLAP':
            sto_model.zero_grad()
            post_model = deepcopy(sto_model)
        if run != 'sketch':
            for _ in range(inner_epochs):
                for start in range(0, len(train_ind), n_train):
                    mbinds = train_ind[start:start + n_train]
                    if run=='Entropy-SGD':
                        train_on_batch_entropy_sgd(x_all[mbinds], y_all[mbinds])
                    elif run=='MLAP':
                        train_on_batch_mlap(x_all[mbinds], y_all[mbinds], post_model)
                    else:
                        train_on_batch(x_all[mbinds], y_all[mbinds])
        if run == 'sketch':
            pass
        elif run == 'MAML':
            outer_step_size = outer_stepsize_maml * (1 - iteration / n_iterations)  # linear schedule
            for start in range(0, len(val_ind), n_train):
                dpinds = val_ind[start:start + n_train]
                x = to_torch(x_all[dpinds])
                y = to_torch(y_all[dpinds])

                # Compute the grads
                model.zero_grad()
                y_pred = model(x)
                loss = (y_pred - y).pow(2).mean()
                loss.backward()

                # Reload the model
                model.load_state_dict(weights_before)

                # SGD on the params
                for param in model.parameters():
                    param.data -= outer_step_size * param.grad.data
        elif run=='Entropy-SGD':
            mu_after = dict()
            for name,params in model.named_parameters():
                mu_after[name] = params.grad
                # print(params.grad)
            # mu_after = model.named_parameters()
            for name,params in model.named_parameters():
                # print(params.grad)
                break
            outerstepsize = outer_stepsize_reptile * 1
            if iteration==0:
                model.load_state_dict({name: mu_after[name] for name in weights_before})
            else:
                model.load_state_dict({name: weights_before[name]*(1-1/(iteration+1)) + mu_after[name]/(1+iteration)
                                    for name in weights_before})
            # print('asdasdasd')
            
            # for name in model.state_dict():
            #     print(model.state_dict()[name])
            #     break
            # prior_param = optimizer.params_group[0]
        elif run == 'MLAP':
            #for param in sto_model.parameters():
            #    param.data -= outer_stepsize_mlap * param.grad.data
            pass
        else:
            # Interpolate between current weights and trained weights from this task
            # I.e. (weights_before - weights_after) is the meta-gradient
            weights_after = model.state_dict()
            outerstepsize = outer_stepsize_reptile * (1 - iteration / n_iterations)  # linear schedule
            model.load_state_dict({name: weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize
                                   for name in weights_before})

        # Periodically plot the results on a particular task and minibatch
        if plot and iteration == 0 or (iteration + 1) % 10 == 0:
            plt.cla()
            f = f_plot
            for f in f_plot_list:
                if run=='MLAP':
                    post_model = deepcopy(sto_model)
                    # post_model = get_model(prm)
                    # weights_before = deepcopy(sto_model.state_dict())
                    # post_model.load_state_dict(weights_before)
                else:
                    post_model = None
                # sketch_model = 
                    weights_before = deepcopy(model.state_dict())  # save snapshot before evaluation
                plt.plot(x_all, predict(x_all, post_model), label="pred after 0", color=(0, 0, 1))
            # print('==============================')
                for inneriter in range(32):
                    # train_on_batch(xtrain_plot, f(xtrain_plot))
                    if run=='Entropy-SGD':
                        # train_on_batch(xtrain_plot, f(xtrain_plot))
                        train_on_batch_entropy_sgd(xtrain_plot, f(xtrain_plot))
                    elif run == 'MLAP':
                        train_on_batch_mlap(xtrain_plot, f(xtrain_plot), post_model, False)
                    else:
                        #train_on_batch_entropy_sgd(xtrain_plot, f(xtrain_plot))
                        train_on_batch(xtrain_plot, f(xtrain_plot))
                    if (inneriter + 1) % 8 == 0:
                        frac = (inneriter + 1) / 32
                        #if run != 'MLAP':
                        #    post_model=None
                        # print((1+iteration)//10)
                        # print((inneriter+1)/8)
                        result[(inneriter+1)//8-1, (1+iteration)//10] += np.square(predict(x_all, post_model)-f(x_all)).mean()/n_test
                        plt.plot(x_all, predict(x_all, post_model), label="pred after %i" % (inneriter + 1), color=(frac, 0, 1 - frac))
                plt.plot(x_all, f(x_all), label="true", color=(0, 1, 0))
                lossval = np.square(predict(x_all, post_model) - f(x_all)).mean()
                # result[iteration] = lossval
                plt.plot(xtrain_plot, f(xtrain_plot), "x", label="train", color="k")
                plt.ylim(-4, 4)
                plt.legend(loc="lower right")
                plt.pause(0.01)
                plt.savefig('results/reptile.jpg')
                if run=='MLAP':
                    sto_model.load_state_dict(weights_before)
                else:
                    model.load_state_dict(weights_before)  # restore from snapshot
                print(f"-----------------------------")
                print(f"iteration               {iteration + 1}")
                print(f"loss on plotted curve   {lossval:.3f}")  # would be better to average loss over a set of examples, but this is optimized for brevity
    # np.savetxt("sketch-ent.txt", result)

def main():
    parser = argparse.ArgumentParser(description='MAML and Reptile Sine wave regression example.')
    parser.add_argument('--run', dest='run', default='MLAP') # MAML, Reptile, Entropy-SGD
    args = parser.parse_args()

    experiment(args.run)



if __name__ == '__main__':
    main()
