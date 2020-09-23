import torch
from torch import nn

class MNIST_CNN(nn.Module):
    """
    A very basic, "naive CNN" model for classifying 8x8 hand-drawn MNIST digits.
    """
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.l1 = nn.Conv2d(1, 10, kernel_size=3) # 10 channels. each "channel" (channel like RGB) is a filter, so 10 new grids.
        self.l2 = nn.Conv2d(10, 20, kernel_size=3) # Another convolutional pass
        self.l3 = nn.Linear(16*20, 10) # This input is 2D - batch number, then 1x16x20, all 20 channels together in a line
        # TODO do i softmax it? I guess I should lol

    def forward(self, x):
        r2 = self.l2(self.l1(x))
        flt = r2.flatten(start_dim=1)
        return self.l3(flt)
