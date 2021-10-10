from torch import nn
from torch.nn import functional as F
import torch

class NLBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(NLBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=(1,1))

        if bn_layer:
            self.W_z = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=(1,1)),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=(1,1))
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=(1,1))
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=(1,1))

    def forward(self, x):
        batch_size = x.size(0)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        return f_div_C
