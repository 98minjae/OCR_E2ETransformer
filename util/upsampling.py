import torch.nn as nn
import torch.nn.functional as F
import torch

class Upsampling(nn.Module):

    def __init__(self):
        super(Upsampling,self).__init__()
        self.mergeLayers0 = DummyLayer()

        self.mergeLayers1 = HLayer(2048 + 1024, 128)
        self.mergeLayers2 = HLayer(128 + 512, 64)
        self.mergeLayers3 = HLayer(64 + 256, 32)

        self.mergeLayers4 = nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
        self.bn5 = nn.BatchNorm2d(32, momentum=0.003)

    def forward(self, f):
        _,_,h,w = f[0].size()
        torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)
        a = nn.Upsample(size = (2*h,2*w),mode = 'bilinear',align_corners = True)
        b = nn.Upsample(size = (4*h,4*w),mode = 'bilinear',align_corners = True)
 
        f[2] = a(f[2])
        f[3] = b(f[3])
        g = [None,None,None,None]
        h = [None,None,None,None]

        # i = 1
        h[0] = self.mergeLayers0(f[0])
        g[0] = h[0]

        # i = 2
        h[1] = self.mergeLayers1(g[0], f[1])
        g[1] = self.__unpool(h[1])

        # i = 3
        h[2] = self.mergeLayers2(g[1], f[2])
        g[2] = self.__unpool(h[2])

        # Match size using Zero-padding

        # i = 4
        h[3] = self.mergeLayers3(g[2], f[3])

        # final stage
        final = self.mergeLayers4(h[3])
        final = self.bn5(final)
        final = F.relu(final)

        return final


    def __unpool(self, input):
        _, _, H, W = input.shape
        return F.interpolate(input, mode = 'bilinear', scale_factor = 2, align_corners = True)


class DummyLayer(nn.Module):
    def __init__(self):
      super(DummyLayer,self).__init__()

    def forward(self, input_f):
        return input_f


class HLayer(nn.Module):

    def __init__(self, inputChannels, outputChannels):
 
        super(HLayer, self).__init__()

        self.conv2dOne = nn.Conv2d(inputChannels, outputChannels, kernel_size = 1)
        self.bnOne = nn.BatchNorm2d(outputChannels, momentum=0.003)

        self.conv2dTwo = nn.Conv2d(outputChannels, outputChannels, kernel_size = 3, padding = 1)
        self.bnTwo = nn.BatchNorm2d(outputChannels, momentum=0.003)

    def forward(self, inputPrevG, inputF):

        input = torch.cat([inputPrevG, inputF], dim = 1)
        output = self.conv2dOne(input)
        output = self.bnOne(output)
        output = F.relu(output)

        output = self.conv2dTwo(output)
        output = self.bnTwo(output)
        output = F.relu(output)

        return output

