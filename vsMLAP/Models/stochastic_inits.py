

from __future__ import absolute_import, division, print_function
import torch

import math

# -----------------------------------------------------------------------------------------------------------#
# Inits
# -----------------------------------------------------------------------------------------------------------#
'''   Xavier initialization
Like in PyTorch's default initializer'''

def init_stochastic_conv2d(m, log_var_init):
    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
    stdv = math.sqrt(1. / n)
    m.w_mu.data.uniform_(-stdv, stdv)
    if m.use_bias:
        m.b_mu.data.uniform_(-stdv, stdv)
        # m.b_log_var.data.normal_(log_var_init['mean'], log_var_init['std'])
        m.b_log_var.data.uniform_(log_var_init['mean'], log_var_init['mean'])
    m.w_log_var.data.uniform_(log_var_init['mean'], log_var_init['mean'])
    # m.w_log_var.data.normal_(log_var_init['mean'], log_var_init['std'])

def init_stochastic_linear(m, log_var_init):
    n = m.w_mu.size(1)
    stdv = math.sqrt(1. / n)
    m.w_mu.data.uniform_(-stdv, stdv)
    if m.use_bias:
        m.b_mu.data.uniform_(-stdv, stdv)
        # m.b_log_var.data.normal_(log_var_init['mean'], log_var_init['std'])
        m.b_log_var.data.uniform_(log_var_init['mean'], log_var_init['mean'])
    m.w_log_var.data.uniform_(log_var_init['mean'], log_var_init['mean'])
    # m.w_log_var.data.normal_(log_var_init['mean'], log_var_init['std'])