
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch
_tokenizer = _Tokenizer()

import numpy                   as np
from   numpy.random            import RandomState
from   torch.autograd           import Variable
from   torch.nn.parameter       import Parameter
from   torch.nn                 import Module
#from   quaternion_ops          import *
import math
import sys
# from fvcore.nn import FlopCountAnalysis

##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

import numpy as np
import pdb
from scipy.stats import chi

def quaternion_linear(input, i_weight, j_weight, bias=True):
    """
    Applies a quaternion linear transformation to the incoming data:

    It is important to notice that the forward phase of a QNN is defined
    as W * Inputs (with * equal to the Hamilton product). The constructed
    cat_kernels_4_quaternion is a modified version of the quaternion representation
    so when we do torch.mm(Input,W) it's equivalent to W * Inputs.

    """


    cat_kernels_4_i = torch.cat([i_weight, j_weight], dim=0)
    cat_kernels_4_j = torch.cat([j_weight, -i_weight], dim=0)

    cat_kernels_4_quaternion   = torch.cat([cat_kernels_4_i, cat_kernels_4_j], dim=1)

    if input.dim() == 2 :

        if bias is not None:
            return torch.addmm(bias, input, cat_kernels_4_quaternion)
        else:
            return torch.mm(input, cat_kernels_4_quaternion)
    else:
        # print('111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
        # print('input',input.shape)
        # print('cat_kernels_4_quaternion', cat_kernels_4_quaternion.shape)
        output = torch.matmul(input, cat_kernels_4_quaternion)
        if bias is not None:
            return output+bias
        else:
            return output


def quaternion_linear_rotation(input, zero_kernel, r_weight, i_weight, j_weight, k_weight, bias=None,
                               quaternion_format=False, scale=None):
    """
    Applies a quaternion rotation transformation to the incoming data:

    The rotation W*x*W^t can be replaced by R*x following:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Works for unitary and non unitary weights.

    The initial size of the input must be a multiple of 3 if quaternion_format = False and
    4 if quaternion_format = True.
    """

    square_r          = (r_weight*r_weight)
    square_i          = (i_weight*i_weight)
    square_j          = (j_weight*j_weight)
    square_k          = (k_weight*k_weight)

    norm              = torch.sqrt(square_r+square_i+square_j+square_k + 0.0001)

    r_n_weight          = (r_weight / norm)
    i_n_weight          = (i_weight / norm)
    j_n_weight          = (j_weight / norm)
    k_n_weight          = (k_weight / norm)

    norm_factor       = 2.0

    square_i          = norm_factor*(i_n_weight*i_n_weight)
    square_j          = norm_factor*(j_n_weight*j_n_weight)
    square_k          = norm_factor*(k_n_weight*k_n_weight)

    ri                = (norm_factor*r_n_weight*i_n_weight)
    rj                = (norm_factor*r_n_weight*j_n_weight)
    rk                = (norm_factor*r_n_weight*k_n_weight)

    ij                = (norm_factor*i_n_weight*j_n_weight)
    ik                = (norm_factor*i_n_weight*k_n_weight)

    jk                = (norm_factor*j_n_weight*k_n_weight)

    if quaternion_format:
        if scale is not None:
            rot_kernel_1  = torch.cat([zero_kernel, scale * (1.0 - (square_j + square_k)), scale *(ij-rk), scale *(ik+rj)], dim=0)
            rot_kernel_2  = torch.cat([zero_kernel, scale *(ij+rk), scale *(1.0 - (square_i + square_k)), scale *(jk-ri)], dim=0)
            rot_kernel_3  = torch.cat([zero_kernel, scale *(ik-rj), scale *(jk+ri), scale *(1.0 - (square_i + square_j))], dim=0)
        else:
            rot_kernel_1  = torch.cat([zero_kernel, (1.0 - (square_j + square_k)), (ij-rk), (ik+rj)], dim=0)
            rot_kernel_2  = torch.cat([zero_kernel, (ij+rk), (1.0 - (square_i + square_k)), (jk-ri)], dim=0)
            rot_kernel_3  = torch.cat([zero_kernel, (ik-rj), (jk+ri), (1.0 - (square_i + square_j))], dim=0)

        zero_kernel2  = torch.cat([zero_kernel, zero_kernel, zero_kernel, zero_kernel], dim=0)
        global_rot_kernel = torch.cat([zero_kernel2, rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=1)

    else:
        if scale is not None:
            rot_kernel_1  = torch.cat([scale * (1.0 - (square_j + square_k)), scale *(ij-rk), scale *(ik+rj)], dim=0)
            rot_kernel_2  = torch.cat([scale *(ij+rk), scale *(1.0 - (square_i + square_k)), scale *(jk-ri)], dim=0)
            rot_kernel_3  = torch.cat([scale *(ik-rj), scale *(jk+ri), scale *(1.0 - (square_i + square_j))], dim=0)
        else:
            rot_kernel_1  = torch.cat([1.0 - (square_j + square_k), (ij-rk), (ik+rj)], dim=0)
            rot_kernel_2  = torch.cat([(ij+rk), 1.0 - (square_i + square_k), (jk-ri)], dim=0)
            rot_kernel_3  = torch.cat([(ik-rj), (jk+ri), (1.0 - (square_i + square_j))], dim=0)

        global_rot_kernel = torch.cat([rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=1)


    if input.dim() == 2 :
        if bias is not None:
            return torch.addmm(bias, input, global_rot_kernel)
        else:
            return torch.mm(input, global_rot_kernel)
    else:
        output = torch.matmul(input, global_rot_kernel)
        if bias is not None:
            return output+bias
        else:
            return output

def quaternion_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features
        fan_out         = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState(np.random.randint(1,1234))

    # Generating randoms and purely imaginary quaternions :
    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    modulus = chi.rvs(4,loc=0,scale=s,size=kernel_shape)
    number_of_weights = np.prod(kernel_shape)
    v_i = np.random.uniform(-1.0,1.0,number_of_weights)
    v_j = np.random.uniform(-1.0,1.0,number_of_weights)
    v_k = np.random.uniform(-1.0,1.0,number_of_weights)

    # Purely imaginary quaternions unitary
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2 +0.0001)
        v_i[i]/= norm
        v_j[i]/= norm
        v_k[i]/= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)


    weight_i = modulus * v_i*np.sin(phase)
    weight_j = modulus * v_j*np.sin(phase)


    return (weight_i, weight_j)

def affect_init(i_weight, j_weight, init_func, rng, init_criterion):
    if i_weight.size() != j_weight.size():
         raise ValueError('The real and imaginary weights '
                 'should have the same size . Found: r:'
               ' i:' + str(i_weight.size()) +' j:'
                 + str(j_weight.size()))

    # elif r_weight.dim() != 2:
    #     raise Exception('affect_init accepts only matrices. Found dimension = '
    #                     + str(r_weight.dim()))
    kernel_size = None
    i, j  = init_func(256, 256, rng, kernel_size, init_criterion) #需要确定输入输出的维度
    i, j  = torch.from_numpy(i), torch.from_numpy(j)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)

class QuaternionLinearAutograd(Module):
    r"""Applies a quaternion linear transformation to the incoming data. A custom
    Autograd function is call to drastically reduce the VRAM consumption. Nonetheless, computing
    time is also slower compared to QuaternionLinear().
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None, rotation=False, quaternion_format=True, scale=False):

        super(QuaternionLinearAutograd, self).__init__()
        self.in_features       = in_features//4
        self.out_features      = out_features//4
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        #置0
        #self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        #self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.scale    = scale

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.scale_param  = None

        # if self.rotation:
        #     self.zero_kernel  = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features*4))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0,1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        # winit = {'quaternion': quaternion_init, 'unitary': unitary_init, 'random': random_init}[self.weight_init]
        #只需要尝试quaternion么？ 一个一个实验
        winit = quaternion_init
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.i_weight, self.j_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if self.rotation:
            return quaternion_linear_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias, self.quaternion_format, self.scale_param)
        else:
            return quaternion_linear(input, self.i_weight, self.j_weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', rotation='       + str(self.rotation) \
            + ', seed=' + str(self.seed) + ')'