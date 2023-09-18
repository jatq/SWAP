import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Add(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, y):
        return x.add(y)
    
    def __repr__(self): 
        return f'{self.__class__.__name__}'
    
class Squeeze(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        
    def forward(self, x): 
        return x.squeeze(self.dim)
    
class Concat(nn.Module):
    def __init__(self, dim=1): 
        super().__init__()
        self.dim = dim
        
    def forward(self, *x): 
        return torch.cat(*x, dim=self.dim)
    
    def __repr__(self): 
        return f'{self.__class__.__name__}(dim={self.dim})'

# Convolution with Batchnorm and ReLU
class ConvBlock(nn.Module):
    def __init__(self, ni, nf, kernel_size=1, stride=1, act = True, bias=None, **kwargs):
        super().__init__()
        conv = nn.Conv1d(ni, nf, kernel_size, bias=bias, stride=stride, padding=kernel_size//2, **kwargs)
        bn = nn.BatchNorm1d(nf)
        self.conv =  nn.Sequential(*[conv, bn, nn.ReLU()]) if act else nn.Sequential(*[conv, bn])

    def forward(self, x):
        return self.conv(x) 


class LKC(nn.Module):
    def __init__(self, dim, k1, k2, bias):
        super().__init__() 
        self.depthwise_conv = nn.Conv1d(dim, dim, k1, stride = 1, padding="same", groups=dim, bias=bias)
        self.depthwise_dila_conv = nn.Conv1d(dim, dim, k2, stride=1, padding = "same", groups=dim, dilation=(k1+1)//2, bias=bias)
        self.pointwise_conv = nn.Conv1d(dim, dim, 1, bias=bias)
        
    def forward(self, x):
        return self.pointwise_conv(
                    self.depthwise_dila_conv(
                        self.depthwise_conv(x)))


class Inception(nn.Module):
    # ni -> nf * 4
    def __init__(self, ni, nf, ks1, ks2, bias, pool_ks = 3):
        super().__init__()
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=bias) 
        self.convs = nn.ModuleList([LKC(dim = nf, k1 = k, k2 = ks2, bias=bias) for k in ks1])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(pool_ks, stride=1, padding=pool_ks//2), nn.Conv1d(ni, nf, 1, bias=bias)])
        self.concat = Concat()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.concat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)])
        return x

    
class Inceptions(nn.Module):
    def __init__(self, ni, nf, ks1, ks2, pool_ks=3):
        super().__init__()
        hidden_channels = [nf * 2**i for i in range(4)]
        out_channels = [h * 4 for h in hidden_channels]
        in_channels = [ni] + out_channels[:-1]


        self.inception_list, self.shortcuts = nn.ModuleList(), nn.ModuleList()
        for i in range(4):
            if (i+1) % 2 == 0:   # when i is 1 or 3
                self.shortcuts.append(ConvBlock(in_channels[i-1],out_channels[i], 1, act=None))
            self.inception_list.append(Inception(in_channels[i], hidden_channels[i], ks1, ks2, bias=False, pool_ks=pool_ks))    
        self.add = Add()
        self.act = nn.ReLU()
        
    def forward(self, x):
        res = x
        for i in range(4):
            x = self.inception_list[i](x)
            if (i + 1) % 2 == 0: 
                res = x = self.act(self.add(x, self.shortcuts[i//2](res)))
        return x
    
class SWAP(nn.Module):
    def __init__(self, c_in=3, c_out=360, nf=32, adaptive_size=25, ks1=[17, 13, 9], ks2=7, pool_ks=3):
        super().__init__()
        self.backbone = Inceptions(c_in, nf, ks1, ks2, pool_ks=pool_ks)
        self.head_nf = nf * 32
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(adaptive_size), 
                                  ConvBlock(self.head_nf, self.head_nf//2, 1), 
                                  ConvBlock(self.head_nf//2, self.head_nf//4, 1), 
                                  ConvBlock(self.head_nf//4, c_out, 1), 
                                  nn.AdaptiveAvgPool1d(1),
                                  Squeeze(-1))

    def forward(self, x, is_softmax=False):
        x = self.backbone(x)
        logits = self.head(x)
        if is_softmax:
            return nn.Softmax(dim=1)(logits)
        return logits
    
    def train_loss(self, x, y):
        pred_y = self.forward(x)
        return  F.kl_div(F.log_softmax(pred_y, dim=1), y, reduction='batchmean')  # log_softmax(logits), probability
    
    def inference(self, x):
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            pred_y = self.forward(x.cuda(), is_softmax=True).cpu().numpy()
            pred_v = np.array([self.label2azimuth(y) for y in pred_y])
        return pred_v
            
    def label2azimuth(self, y):
        # transform a probability distribution to azimuth value.
        sep = 360//y.shape[-1]
        i = np.arange(0, 360, sep)
        index = np.arange(0,int(360/sep)).astype(np.int32)
        max_i = i[np.argmax(y)]
        if abs(max_i - 180) > 100:
            i = np.arange(-180, 180, sep)
            index = np.arange(int(-180/sep), int(180/sep)).astype(np.int32)
        return np.sum(y[index] * i) % 360 %360
        

if __name__ == '__main__':
    m = SWAP(3, 360, nf=32,  ks1 = [17, 11, 5], ks2=7).cuda()
    x = torch.randn(10, 3, 200)
    print(m.inference(x))
