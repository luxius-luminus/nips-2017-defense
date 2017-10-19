import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Transformer(nn.Module):
    def __init__(self, ngf=64, dropout=False):
        super(Transformer, self).__init__()
        en_factr = [1, 2, 4, 8, 8, 8, 8, 8]
        de_factr = [8, 8, 8, 8, 8, 4, 2, 1]
        model = [ConvModule(3, ngf*en_factr[0], kernel_size=3, stride=2,
                            padding=1)]
        for i in range(1, len(en_factr)):
            model.append(ConvModule(ngf*en_factr[i-1],
                                    ngf*en_factr[i],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1))

        for i in range(len(de_factr)-1):
            if i == 2:
                model.append(ConvTransposeModule(ngf*de_factr[i],
                                        ngf*de_factr[i+1],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1))
            else:
                model.append(ConvTransposeModule(ngf*de_factr[i],
                                        ngf*de_factr[i+1],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1))
            if dropout and i < 2:
                model.append(nn.Dropout(0.5))

        model.append(LastConvTransposeModule(ngf*de_factr[len(de_factr)-1], 3,
                                kernel_size=3, stride=2,
                                padding=1,
                                output_padding=1))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Transformer2(nn.Module):
    def __init__(self, ngf=64, dropout=False):
        super(Transformer2, self).__init__()
        en_factr = [1, 2, 4, 8]
        de_factr = [8, 4, 2, 1]
        model = [ConvModule(3, ngf*en_factr[0], kernel_size=3, stride=2,
                            padding=1)]
        for i in range(1, len(en_factr)):
            model.append(ConvModule(ngf*en_factr[i-1],
                                    ngf*en_factr[i],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1))

        for i in range(len(de_factr)-1):
            model.append(ConvTransposeModule(ngf*de_factr[i],
                                    ngf*de_factr[i+1],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1))

        model.append(LastConvTransposeModule(ngf*de_factr[len(de_factr)-1], 3,
                                kernel_size=3, stride=2,
                                padding=1,
                                output_padding=1))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Transformer3(nn.Module):
    def __init__(self, ngf=64, dropout=False):
        super(Transformer3, self).__init__()
        en_factr = [1, 2, 4, 8]
        de_factr = [8, 4, 2, 1]
        model = [ConvModule(3, ngf*en_factr[0], kernel_size=3, stride=2,
                            padding=1)]
        for i in range(1, len(en_factr)):
            model.append(ConvModule(ngf*en_factr[i-1],
                                    ngf*en_factr[i],
                                    kernel_size=3,
                                    stride=2))

        for i in range(len(de_factr)-1):
            model.append(ConvTransposeModule(ngf*de_factr[i],
                                    ngf*de_factr[i+1],
                                    kernel_size=3,
                                    stride=2,
                                    output_padding=1))

        model.append(LastConvTransposeModule(ngf*de_factr[len(de_factr)-1], 3,
                                kernel_size=3, stride=2,
                                padding=1,
                                ))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Transformer4(nn.Module):
    def __init__(self, ngf=64, dropout=False):
        super(Transformer4, self).__init__()
        en_factr = [1, 2, 4, 8]
        de_factr = [8, 4, 2, 1]
        model = [ConvModule(3, ngf*en_factr[0], kernel_size=3, stride=2,
                            padding=1)]
        for i in range(1, len(en_factr)):
            model.append(ConvModule(ngf*en_factr[i-1],
                                    ngf*en_factr[i],
                                    kernel_size=3,
                                    stride=2, padding=1))

        output_padding = [1, 0, 1]
        for i in range(len(de_factr)-1):
            model.append(ConvTransposeModule(ngf*de_factr[i],
                                    ngf*de_factr[i+1],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=output_padding[i]))

        model.append(LastConvTransposeModule(ngf*de_factr[len(de_factr)-1], 3,
                                kernel_size=3, stride=2,
                                padding=1,
                                ))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
class ConvModule(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size=3, **kargs):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_plane, out_plane, kernel_size, **kargs)
        self.bn = nn.BatchNorm2d(out_plane)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ConvTransposeModule(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size=3, **kargs):
        super(ConvTransposeModule, self).__init__()
        self.conv = nn.ConvTranspose2d(in_plane, out_plane, kernel_size, **kargs)
        self.bn = nn.BatchNorm2d(out_plane)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class LastConvTransposeModule(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size=3, **kargs):
        super(LastConvTransposeModule, self).__init__()
        self.conv = nn.ConvTranspose2d(in_plane, out_plane, kernel_size, **kargs)
        self.bn = nn.BatchNorm2d(out_plane)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.tanh(out)
        return out
