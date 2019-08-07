import torch
import pdb
import traceback
import time
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from torch import autograd


class GuruMeditation (autograd.detect_anomaly):  
    def __init__(self):
        super(GuruMeditation, self).__init__()

    def __enter__(self):
        super(GuruMeditation, self).__enter__()
        return self  

    def __exit__(self, type, value, trace):
        super(GuruMeditation, self).__exit__()
        if isinstance(value, RuntimeError):
            traceback.print_tb(trace)
            Halt(str(value))

def Halt(msg):
    print (msg)
    pdb.set_trace()


def PlotGradFlow(named_parameters, dir, epoch, prefix=''):
    '''
    Plots the gradient by layer and saves it to the figure. 
    In order to make it more accomodating for large values, we plot
    y = OFFSET(20) + log(grad)
    So infer results accordingly. Use as,
        loss.backward()
        PlotGradFlow(model.named_parameters(),...)
    '''
    ave_grads = []
    max_grads = []
    OFF = 20.0
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            if p.grad is None:
                ave_grads.append(0)
                max_grads.append(0)
            else:
                #print (p.grad.detach().cpu().numpy())
                ave_grads.append(max(OFF + np.log(abs(p.grad.detach().cpu().numpy()).mean() + 1e-20), 0))
                max_grads.append(max(OFF + np.log(abs(p.grad.detach().cpu().numpy()).max() + 1e-20), 0))
    plt.figure(figsize=(18, 24))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(
        range(0, len(ave_grads), 1),
        layers,
        rotation="vertical",
        fontsize='x-small'
    )
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-1e-6, top=np.array(ave_grads).max())  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    curr_time = time.time()
    plt.savefig(os.path.join(dir, prefix+str(epoch)+'_'+str(int(curr_time)) + '.png'), dpi=300)
    plt.close('all')
