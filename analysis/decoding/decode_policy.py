import torch
from neurotools import util


class MapDecode(torch.nn.Module):

    @staticmethod
    def linear_act(x):
        return x

    def __init__(self, shape, dev='cuda', out=7, nonlinear=True):
        super().__init__()
        self.shape = shape
        self.pool_f2 = torch.nn.AvgPool3d(2)
        if nonlinear:
            self.activ = torch.nn.ReLU()
        else:
            self.activ = self.linear_act

        kernel, pad = util.conv_identity_params(in_spatial=64, desired_kernel=3)
        self.conv1 = torch.nn.Conv3d(kernel_size=kernel, padding=pad, in_channels=1, out_channels=3, device=dev)
        self.bn1 = torch.nn.BatchNorm3d(3, device=dev)

        kernel, pad = util.conv_identity_params(in_spatial=32, desired_kernel=3)
        self.conv2 = torch.nn.Conv3d(kernel_size=kernel, padding=pad, in_channels=3, out_channels=4, device=dev)
        self.bn2 = torch.nn.BatchNorm3d(4, device=dev)

        kernel, pad = util.conv_identity_params(in_spatial=16, desired_kernel=3)
        self.conv3 = torch.nn.Conv3d(kernel_size=kernel, padding=pad, in_channels=4, out_channels=5, device=dev)
        self.bn3 = torch.nn.BatchNorm3d(5, device=dev)

        kernel, pad = util.conv_identity_params(in_spatial=8, desired_kernel=3)
        self.conv4 = torch.nn.Conv3d(kernel_size=kernel, padding=pad, in_channels=5, out_channels=6, device=dev)
        self.bn4 = torch.nn.BatchNorm3d(6, device=dev)

        kernel, pad = util.conv_identity_params(in_spatial=4, desired_kernel=3)
        self.conv5 = torch.nn.Conv3d(kernel_size=kernel, padding=pad, in_channels=6, out_channels=out, device=dev)
        self.bn5 = torch.nn.BatchNorm3d(out, device=dev)

        self.pool_out = torch.nn.AvgPool3d(2)

    def requires_grad_(self, requires_grad: bool = True):
        for param in self.parameters():
            param.data.requires_grad_(requires_grad)

    def to(self, device):
        self.conv1 = self.conv1.to(device)
        self.bn1 = self.bn1.to(device)
        self.conv2 = self.conv2.to(device)
        self.bn2 = self.bn2.to(device)
        self.conv3 = self.conv3.to(device)
        self.bn3 = self.bn3.to(device)
        self.conv4 = self.conv4.to(device)

        self.bn4 = self.bn4.to(device)
        self.conv5 = self.conv5.to(device)
        self.bn5 = self.bn5.to(device)
        return self

    def norm(self, x):
       #  std = torch.std(x, dim=(0, 2, 3, 4))
        mean = torch.mean(x, dim=(0, 2, 3, 4))
        h = (x - mean.view((1, len(mean), 1, 1, 1))) # / std.view((1, len(std), 1, 1, 1))
        return h

    def forward(self, x):
        h = self.conv1(x)
        h = self.pool_f2(h)
        h = self.bn1(h)
        h = self.activ(h)

        h = self.conv2(h)
        h = self.pool_f2(h)
        h = self.bn2(h)
        h = self.activ(h)

        h = self.conv3(h)
        h = self.pool_f2(h)
        h = self.bn3(h)
        h = self.activ(h)

        h = self.conv4(h)
        h = self.pool_f2(h)
        h = self.bn4(h)
        h = self.activ(h)

        h = self.conv5(h)
        h = self.pool_f2(h)
        h = self.bn5(h)
        h = self.activ(h)

        h = self.pool_out(h)
        h = self.norm(h)

        yhat = h.squeeze()
        return yhat


class LinearDecode(torch.nn.Module):

    def __init__(self, defualt_mask, in_channels=2, out=7, dev='cuda'):
        super().__init__()
        self.mask = defualt_mask
        self.device = dev
        self.in_size = torch.count_nonzero(defualt_mask).detach().cpu().item() * in_channels
        self.weights = torch.nn.Linear(in_features=self.in_size, out_features=out, device=self.device)

    def forward(self, x):
        x = x[:, :, self.mask]
        x = x.reshape((x.shape[0], -1))
        yhat = self.weights(x)
        return yhat

    def spatial_weights(self):
        """
        Returns
        -------
        Weight matrix in the original dimensional space.
        """
        space = torch.zeros(size=self.mask.shape)
        space[self.mask] = self.weights
        return self.weights

