import torch
import torch.nn.functional as F


class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

    def backward(self, x):
        sigmoid = torch.sigmoid(x)
        tanh_sigmoid = torch.tanh(F.softplus(x))

        grad_mish = x * (tanh_sigmoid + x * (1 - tanh_sigmoid ** 2) * sigmoid)
        return grad_mish
    


