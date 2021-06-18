import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import higher

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
set_seed(100)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

#inner var
model = MyModel()
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0)

#outer var
weight = torch.Tensor([1])
weight.requires_grad = True

#train data
xt = torch.Tensor([1])
yt = torch.Tensor([1])

#val data
xv = torch.Tensor([2])
yv = torch.Tensor([2])

with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
    logit_t = fmodel(xt)
    loss_t = weight*torch.pow(logit_t-yt, 2)
    diffopt.step(loss_t)

    logit_v = fmodel(xv)
    loss_v = torch.pow(logit_v-yv, 2)
    grad = torch.autograd.grad(loss_v, weight)[0].data

    weight = weight - 0.1*grad

