import torch
import numpy as np
from torch import autograd
import pdb
import traceback
import torch.nn as nn
import torch.optim as optim

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
            halt(str(value))

def halt(msg):
    print (msg)
    pdb.set_trace()


class MyFunc(autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        return inp.clone()
    @staticmethod
    def backward(ctx, gO):
        # Error during the backward pass
        raise RuntimeError("Some error in backward")
        return gO.clone()


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # RGB image and a mask...
        self.fc3 = nn.Linear(10, 15, bias=False)
    
    def forward(self, x) :
        x2 = self.fc3(x)
        x2[0] = 1/0.


encoderInit = Encoder()
opEncoderInit = optim.Adam(encoderInit.parameters(), lr=1e-3 * scale, betas=(0.5, 0.999))

def run_fn(a):
    #out = MyFunc.apply(a)
    out = encoderInit(a)
    return out.sum()

with GuruMeditation() as gr :
    inp = torch.rand(10, 10, requires_grad=True)
    out = run_fn(inp)
    out.backward()