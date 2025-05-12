from os.path import join
from tqdm import tqdm
import pandas as pd

import torchvision
import torch
import torch.nn as nn


class Conv2dTanh(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), bias=True):
        super(Conv2dTanh, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        """
        Args:
            x: [N,C,H,W]
        """
        o1 = self.conv(x)
        o2 = torch.tanh(o1)
        return o2


class Features(nn.Module):
    def __init__(self, padding):
        super(Features, self).__init__()

        self.padding = padding

        self.conv2dtanh1 = Conv2dTanh(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2dtanh2 = Conv2dTanh(in_channels=6, out_channels=16, kernel_size=5, padding=2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.conv2dtanh3 = Conv2dTanh(in_channels=16, out_channels=120, kernel_size=5, padding=2)

    def forward(self, x):
        """
        Args:
            x: [N,1,H,W]
        """
        o1 = self.conv2dtanh1(x)
        o2 = self.avgpool1(o1)
        o3 = self.conv2dtanh2(o2)
        o4 = self.avgpool2(o3)
        o5 = self.conv2dtanh3(o4)
        return o5


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()

        self.num_classes = num_classes

        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        """
        Args:
            x: [N,120]
        """
        o1 = self.fc1(x)
        o2 = torch.tanh(o1)
        o3 = self.fc2(o2)
        return o3


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, padding=0):
        super(LeNet5, self).__init__()

        self.num_classes = num_classes
        self.padding = padding

        self.features = Features(padding=padding)
        self.flatten = nn.Flatten()
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, x):
        """
        Args:
            x: [N,1,H,W]
        """
        o1 = self.features(x)
        o2 = self.flatten(o1)
        o3 = self.classifier(o2)
        o4 = torch.log_softmax(o3, dim=-1)
        return o4


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=embedding_dim)
        self.fc2 = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

    def forward(self, x):
        """
        Args:
            x: [N,L]
        """
        o1 = self.embedding(x)
        o2 = self.lstm(o1)
        o3 = self.fc1(o2[0])
        o4 = self.fc2(o3)
        o5 = torch.log_softmax(o4, dim=-1)
        return o5


class SubModel6(nn.Module):
    def __init__(self, name, num_classes=15, hidden_size=256, in_features=50, out_features=50, bias=True, shape=(1, 10)):
        super(SubModel6, self).__init__()

        self.name = name
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.shape = shape

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=256, out_features=256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        """
        Args:
            x: [500,256]
        """
        o1 = self.flatten(x)
        o2 = self.fc1(o1)
        o3 = self.relu(o2)
        o4 = self.fc2(o3)
        return o4


class Conv2dTanh2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), bias=True):
        super(Conv2dTanh2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        """
        Args:
            x: [N,C,H,W]
        """
        o1 = self.conv(x)
        o2 = torch.tanh(o1)
        return o2


class MotorControl(nn.Module):
    def __init__(self, num_classes, hidden_size=256, inplace=False):
        super(MotorControl, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.inplace = inplace

        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        """
        Args:
            x: [N,256]
        """
        o1 = self.fc1(x)
        o2 = self.relu(o1)
        o3 = self.fc2(o2)
        o4 = self.fc3(o3)
        o5 = torch.sigmoid(o4)
        o6 = torch.tanh(o4)
        o7 = torch.add(o5, other=o6)
        return o7


class multihiddenlayer(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(multihiddenlayer, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=20, out_features=20)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=20, out_features=20)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=20, out_features=20)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(in_features=20, out_features=20)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(in_features=20, out_features=20)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(in_features=20, out_features=20)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(in_features=20, out_features=20)
        self.relu8 = nn.ReLU()
        self.fc9 = nn.Linear(in_features=20, out_features=20)
        self.relu9 = nn.ReLU()
        self.fc10 = nn.Linear(in_features=20, out_features=20)
        self.relu10 = nn.ReLU()
        self.fc11 = nn.Linear(in_features=20, out_features=20)
        self.relu11 = nn.ReLU()
        self.fc12 = nn.Linear(in_features=20, out_features=20)
        self.relu12 = nn.ReLU()
        self.fc13 = nn.Linear(in_features=20, out_features=20)
        self.relu13 = nn.ReLU()
        self.fc14 = nn.Linear(in_features=20, out_features=20)
        self.relu14 = nn.ReLU()
        self.fc15 = nn.Linear(in_features=20, out_features=20)
        self.relu15 = nn.ReLU()
        self.fc16 = nn.Linear(in_features=20, out_features=20)
        self.relu16 = nn.ReLU()
        self.fc17 = nn.Linear(in_features=20, out_features=20)
        self.relu17 = nn.ReLU()
        self.fc18 = nn.Linear(in_features=20, out_features=20)
        self.relu18 = nn.ReLU()
        self.fc19 = nn.Linear(in_features=20, out_features=20)
        self.relu19 = nn.ReLU()
        self.fc20 = nn.Linear(in_features=20, out_features=20)
        self.relu20 = nn.ReLU()
        self.fc21 = nn.Linear(in_features=20, out_features=20)
        self.relu21 = nn.ReLU()
        self.fc22 = nn.Linear(in_features=20, out_features=20)
        self.relu22 = nn.ReLU()
        self.fc23 = nn.Linear(in_features=20, out_features=20)
        self.relu23 = nn.ReLU()
        self.fc24 = nn.Linear(in_features=20, out_features=20)
        self.relu24 = nn.ReLU()
        self.fc25 = nn.Linear(in_features=20, out_features=20)
        self.relu25 = nn.ReLU()
        self.fc26 = nn.Linear(in_features=20, out_features=20)
        self.relu26 = nn.ReLU()
        self.fc27 = nn.Linear(in_features=20, out_features=20)
        self.relu27 = nn.ReLU()
        self.fc28 = nn.Linear(in_features=20, out_features=20)
        self.relu28 = nn.ReLU()
        self.fc29 = nn.Linear(in_features=20, out_features=20)
        self.relu29 = nn.ReLU()
        self.fc30 = nn.Linear(in_features=20, out_features=10)

    def forward(self, x):
        """
        Args:
            x: [N,784]
        """
        o1 = self.fc1(x)
        o2 = self.relu1(o1)
        o3 = self.fc2(o2)
        o4 = self.relu2(o3)
        o5 = self.fc3(o4)
        o6 = self.relu3(o5)
        o7 = self.fc4(o6)
        o8 = self.relu4(o7)
        o9 = self.fc5(o8)
        o10 = self.relu5(o9)
        o11 = self.fc6(o10)
        o12 = self.relu6(o11)
        o13 = self.fc7(o12)
        o14 = self.relu7(o13)
        o15 = self.fc8(o14)
        o16 = self.relu8(o15)
        o17 = self.fc9(o16)
        o18 = self.relu9(o17)
        o19 = self.fc10(o18)
        o20 = self.relu10(o19)
        o21 = self.fc11(o20)
        o22 = self.relu11(o21)
        o23 = self.fc12(o22)
        o24 = self.relu12(o23)
        o25 = self.fc13(o24)
        o26 = self.relu13(o25)
        o27 = self.fc14(o26)
        o28 = self.relu14(o27)
        o29 = self.fc15(o28)
        o30 = self.relu15(o29)
        o31 = self.fc16(o30)
        o32 = self.relu16(o31)
        o33 = self.fc17(o32)
        o34 = self.relu17(o33)
        o35 = self.fc18(o34)
        o36 = self.relu18(o35)
        o37 = self.fc19(o36)
        o38 = self.relu19(o37)
        o39 = self.fc20(o38)
        o40 = self.relu20(o39)
        o41 = self.fc21(o40)
        o42 = self.relu21(o41)
        o43 = self.fc22(o42)
        o44 = self.relu22(o43)
        o45 = self.fc23(o44)
        o46 = self.relu23(o45)
        o47 = self.fc24(o46)
        o48 = self.relu24(o47)
        o49 = self.fc25(o48)
        o50 = self.relu25(o49)
        o51 = self.fc26(o50)
        o52 = self.relu26(o51)
        o53 = self.fc27(o52)
        o54 = self.relu27(o53)
        o55 = self.fc28(o54)
        o56 = self.relu28(o55)
        o57 = self.fc29(o56)
        o58 = self.relu29(o57)
        o59 = self.fc30(o58)
        return o59


class Speech(nn.Module):
    def __init__(self, num_classes, hidden_size=256, start_dim=1, end_dim=-1, in_features=1, out_features=256, bias=True, inplace=False):
        super(Speech, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.inplace = inplace

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=256, out_features=60)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=60, out_features=30)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=30, out_features=10)

    def forward(self, x):
        """
        Args:
            x: [N,256]
        """
        o1 = self.flatten(x)
        o2 = self.fc1(o1)
        o3 = self.relu1(o2)
        o4 = self.fc2(o3)
        o5 = self.relu2(o4)
        o6 = self.fc3(o5)
        return o6


class Conv2dTanh3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), bias=True):
        super(Conv2dTanh3, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        """
        Args:
            x: [N,C,H,W]
        """
        o1 = self.conv(x)
        o2 = torch.tanh(o1)
        return o2


class Features3(nn.Module):
    def __init__(self, padding):
        super(Features3, self).__init__()

        self.padding = padding

        self.conv2dtanh31 = Conv2dTanh3(in_channels=1, out_channels=6, kernel_size=5, padding=padding)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2dtanh32 = Conv2dTanh3(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.conv2dtanh33 = Conv2dTanh3(in_channels=16, out_channels=120, kernel_size=5)

    def forward(self, x):
        """
        Args:
            x: [N,1,H,W]
        """
        o1 = self.conv2dtanh31(x)
        o2 = self.avgpool1(o1)
        o3 = self.conv2dtanh32(o2)
        o4 = self.avgpool2(o3)
        o5 = self.conv2dtanh33(o4)
        return o5


class Classifier3(nn.Module):
    def __init__(self, num_classes):
        super(Classifier3, self).__init__()

        self.num_classes = num_classes

        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        """
        Args:
            x: [N,120]
        """
        o1 = self.fc1(x)
        o2 = torch.tanh(o1)
        o3 = self.fc2(o2)
        return o3


class LeNet53(nn.Module):
    def __init__(self, num_classes=10, padding=0):
        super(LeNet53, self).__init__()

        self.num_classes = num_classes
        self.padding = padding

        self.features3 = Features3(padding=padding)
        self.flatten = nn.Flatten()
        self.classifier3 = Classifier3(num_classes=num_classes)

    def forward(self, x):
        """
        Args:
            x: [N,1,H,W]
        """
        o1 = self.features3(x)
        o2 = self.flatten(o1)
        o3 = self.classifier3(o2)
        o4 = torch.log_softmax(o3, dim=-1)
        return o4


class Vision(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(Vision, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        o5 = torch.log_softmax(dim=-1)
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.relu8 = nn.ReLU()
        self.relu9 = nn.ReLU()
        self.relu10 = nn.ReLU()
        self.relu11 = nn.ReLU()
        self.relu12 = nn.ReLU()
        self.relu13 = nn.ReLU()
        self.relu14 = nn.ReLU()
        self.relu15 = nn.ReLU()
        self.relu16 = nn.ReLU()
        self.relu17 = nn.ReLU()
        self.relu18 = nn.ReLU()
        self.relu19 = nn.ReLU()
        self.relu20 = nn.ReLU()
        self.relu21 = nn.ReLU()
        self.relu22 = nn.ReLU()
        self.relu23 = nn.ReLU()
        self.relu24 = nn.ReLU()
        self.relu25 = nn.ReLU()
        self.relu26 = nn.ReLU()
        self.relu27 = nn.ReLU()
        self.relu28 = nn.ReLU()
        self.relu29 = nn.ReLU()
        self.relu30 = nn.ReLU()
        self.relu31 = nn.ReLU()
        self.relu32 = nn.ReLU()
        self.relu33 = nn.ReLU()
        self.relu34 = nn.ReLU()
        self.relu35 = nn.ReLU()
        self.relu36 = nn.ReLU()
        self.relu37 = nn.ReLU()
        self.relu38 = nn.ReLU()
        self.relu39 = nn.ReLU()
        self.relu40 = nn.ReLU()
        self.relu41 = nn.ReLU()
        self.relu42 = nn.ReLU()
        self.relu43 = nn.ReLU()
        self.relu44 = nn.ReLU()
        self.relu45 = nn.ReLU()
        self.relu46 = nn.ReLU()
        self.relu47 = nn.ReLU()
        self.relu48 = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [500,256]
        """
        o1 = self.relu1(x)
        o2 = self.relu2(x)
        o3 = self.relu3(x)
        o4 = self.relu4(x)
        o6 = self.relu5(x)
        o7 = self.relu6(x)
        o8 = self.relu7(x)
        o9 = self.relu8(x)
        o10 = torch.tanh(o1)
        o11 = self.relu9(o4)
        o12 = self.relu10(o1)
        o13 = self.relu11(o2)
        o14 = self.relu12(o3)
        o15 = self.relu13(o6)
        o16 = self.relu14(o7)
        o17 = self.relu15(o8)
        o18 = self.relu16(o9)
        o19 = torch.tanh(o9)
        o20 = torch.tanh(o12)
        o21 = self.relu17(o12)
        o22 = self.relu18(o13)
        o23 = self.relu19(o14)
        o24 = self.relu20(o11)
        o25 = self.relu21(o15)
        o26 = self.relu22(o16)
        o27 = self.relu23(o17)
        o28 = self.relu24(o18)
        o29 = torch.tanh(o18)
        o30 = self.relu25(o21)
        o31 = self.relu26(o22)
        o32 = self.relu27(o23)
        o33 = torch.tanh(o21)
        o34 = self.relu28(o24)
        o35 = self.relu29(o25)
        o36 = self.relu30(o26)
        o37 = self.relu31(o27)
        o38 = self.relu32(o28)
        o39 = torch.tanh(o28)
        o40 = torch.tanh(o30)
        o41 = self.relu33(o30)
        o42 = self.relu34(o31)
        o43 = self.relu35(o32)
        o44 = self.relu36(o34)
        o45 = self.relu37(o35)
        o46 = self.relu38(o36)
        o47 = self.relu39(o37)
        o48 = self.relu40(o38)
        o49 = torch.tanh(o38)
        o50 = torch.tanh(o41)
        o51 = self.relu41(o41)
        o52 = self.relu42(o42)
        o53 = self.relu43(o43)
        o54 = self.relu44(o44)
        o55 = self.relu45(o45)
        o56 = self.relu46(o46)
        o57 = self.relu47(o47)
        o58 = self.relu48(o48)
        o59 = torch.tanh(o48)
        o60 = torch.tanh(o51)
        o61 = torch.tanh(o52)
        o62 = torch.tanh(o53)
        o63 = torch.tanh(o54)
        o64 = torch.tanh(o55)
        o65 = torch.tanh(o56)
        o66 = torch.tanh(o57)
        o67 = torch.tanh(o58)
        return o63, o62, o61, o60, o64, o65, o66, o67, o59, o49, o39, o29, o19, o50, o40, o33, o20, o10


class VOLT_A1(nn.Module):
    def __init__(self, num_classes, padding, dim, hidden_size=256, shape=1, keepdim=False, start_dim=1, end_dim=-1, in_features=50, out_features=50):
        super(VOLT_A1, self).__init__()

        self.num_classes = num_classes
        self.padding = padding
        self.dim = dim
        self.hidden_size = hidden_size
        self.shape = shape
        self.keepdim = keepdim
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.in_features = in_features
        self.out_features = out_features

        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()
        self.flatten4 = nn.Flatten()
        self.flatten5 = nn.Flatten()
        self.flatten6 = nn.Flatten()
        self.flatten7 = nn.Flatten()
        self.flatten8 = nn.Flatten()
        self.flatten9 = nn.Flatten()
        self.flatten10 = nn.Flatten()
        self.flatten11 = nn.Flatten()
        self.flatten12 = nn.Flatten()
        self.flatten13 = nn.Flatten()
        self.flatten14 = nn.Flatten()
        self.flatten15 = nn.Flatten()
        self.flatten16 = nn.Flatten()
        self.flatten17 = nn.Flatten()
        self.flatten18 = nn.Flatten()
        self.flatten19 = nn.Flatten()
        self.flatten20 = nn.Flatten()
        self.multihiddenlayer1 = multihiddenlayer(num_classes=10)
        self.multihiddenlayer2 = multihiddenlayer(num_classes=10)
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        self.multihiddenlayer3 = multihiddenlayer(num_classes=10)
        self.multihiddenlayer4 = multihiddenlayer(num_classes=10)
        self.multihiddenlayer5 = multihiddenlayer(num_classes=10)
        self.fc2 = nn.Linear(in_features=784, out_features=256)
        self.multihiddenlayer6 = multihiddenlayer(num_classes=10)
        self.multihiddenlayer7 = multihiddenlayer(num_classes=10)
        self.multihiddenlayer8 = multihiddenlayer(num_classes=10)
        self.multihiddenlayer9 = multihiddenlayer(num_classes=10)
        self.fc3 = nn.Linear(in_features=784, out_features=256)
        self.multihiddenlayer10 = multihiddenlayer(num_classes=10)
        self.fc4 = nn.Linear(in_features=784, out_features=256)
        self.multihiddenlayer11 = multihiddenlayer(num_classes=10)
        self.multihiddenlayer12 = multihiddenlayer(num_classes=10)
        self.multihiddenlayer13 = multihiddenlayer(num_classes=10)
        self.multihiddenlayer14 = multihiddenlayer(num_classes=10)
        self.multihiddenlayer15 = multihiddenlayer(num_classes=10)
        self.multihiddenlayer16 = multihiddenlayer(num_classes=10)
        self.relu1 = nn.ReLU()
        self.submodel6 = SubModel6(out_features=10, name=0)
        self.relu2 = nn.ReLU()
        self.speech = Speech(num_classes=10)
        self.fc5 = nn.Linear(in_features=256, out_features=10)
        self.motorcontrol = MotorControl(num_classes=10)
        self.fc6 = nn.Linear(in_features=10, out_features=10)

    def forward(self, x):
        """
        Args:
            x: [N,1,28,28]
        """
        o1 = self.flatten1(x)
        o2 = self.flatten2(x)
        o3 = self.flatten3(x)
        o4 = self.flatten4(x)
        o5 = self.flatten5(x)
        o6 = self.flatten6(x)
        o7 = self.flatten7(x)
        o8 = self.flatten8(x)
        o9 = self.flatten9(x)
        o10 = self.flatten10(x)
        o11 = self.flatten11(x)
        o12 = self.flatten12(x)
        o13 = self.flatten13(x)
        o14 = self.flatten14(x)
        o15 = self.flatten15(x)
        o16 = self.flatten16(x)
        o17 = self.flatten17(x)
        o18 = self.flatten18(x)
        o19 = self.flatten19(x)
        o20 = self.flatten20(x)
        o21 = self.multihiddenlayer1(o9)
        o22 = self.multihiddenlayer2(o2)
        o23 = self.fc1(o12)
        o24 = self.multihiddenlayer3(o11)
        o25 = self.multihiddenlayer4(o4)
        o26 = self.multihiddenlayer5(o8)
        o27 = self.fc2(o1)
        o28 = self.multihiddenlayer6(o7)
        o29 = self.multihiddenlayer7(o6)
        o30 = self.multihiddenlayer8(o5)
        o31 = self.multihiddenlayer9(o13)
        o32 = self.fc3(o10)
        o33 = self.multihiddenlayer10(o14)
        o34 = self.fc4(o3)
        o35 = self.multihiddenlayer11(o15)
        o36 = self.multihiddenlayer12(o16)
        o37 = self.multihiddenlayer13(o17)
        o38 = self.multihiddenlayer14(o18)
        o39 = self.multihiddenlayer15(o19)
        o40 = self.multihiddenlayer16(o20)
        o41 = self.relu1(o34)
        o42 = self.submodel6(o27)
        o43 = self.relu2(o32)
        o44 = self.speech(o23)
        o45 = torch.add(o29, other=o30)
        o46 = self.fc5(o41)
        o47 = self.motorcontrol(o43)
        o48 = torch.add(o45, other=o31)
        o49 = torch.add(o48, other=o33)
        o50 = torch.add(o49, other=o35)
        o51 = torch.add(o50, other=o36)
        o52 = torch.add(o51, other=o38)
        o53 = torch.add(o52, other=o37)
        o54 = torch.add(o53, other=o39)
        o55 = torch.add(o54, other=o40)
        o56 = self.fc6(o55)
        o57 = torch.log_softmax(o56, dim=-1)
        return o57


class Datasets:
    def __init__(self, dataset_path, batch_size):
        self.train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST(dataset_path, train=True, download=True,
                                     transform=torchvision.transforms.ToTensor()),
          batch_size=batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST(dataset_path, train=False, download=True,
                                     transform=torchvision.transforms.ToTensor()),
          batch_size=batch_size * 2, shuffle=True)


class Trainer:
    def __init__(self, datasets, model, optimizer, loss_fn, results_path='results'):
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.results_path = results_path

        self.train_df = None

    def train_epoch(self, msg_format):
        self.model.train()

        losses = []
        bar = tqdm(self.datasets.train_loader)
        for data, target in bar:
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.loss_fn(output, target)

            loss.backward()
            self.optimizer.step()

            bar.set_description(msg_format.format(loss.item()))

            losses.append(loss.item())
        return losses

    def test(self):
        self.model.eval()

        count = len(self.datasets.test_loader.dataset)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.datasets.test_loader:
                output = self.model(data)
                test_loss += self.loss_fn(output, target).item() * len(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()

        return test_loss / count, correct / count

    def train(self, num_epoch):
        val_loss, accuracy = self.test()
        all_losses = [[None, val_loss, accuracy]]

        for epoch in range(num_epoch):
            # train
            train_losses = self.train_epoch(
                f'train {epoch}/{num_epoch} -- loss: {{:3.2f}}, val_loss: {val_loss:3.2f}, accuracy: {accuracy:.1%}')

            # test
            val_loss, accuracy = self.test()
            all_losses.extend([
                [train_loss, None, None]
                for train_loss in train_losses
            ])
            all_losses.append([None, val_loss, accuracy])

        self.save_model()
        self.train_df = pd.DataFrame(data=all_losses, columns=["train_loss", "val_loss", "accuracy"])
        self.train_df.to_csv(join(self.results_path, "train.csv"), index=False)

    def save_model(self):
        torch.save(self.model.state_dict(), join(self.results_path, 'model.pth'))

    def plot(self):
        import matplotlib.pyplot as plt
        self.train_df[["train_loss", "val_loss"]].ffill().plot(grid=True, logy=True)
        self.train_df[["accuracy"]].dropna().plot(grid=True)
        plt.show()


def train():
    torch.manual_seed(0)
    
    # Enable CUDA optimization
    torch.backends.cudnn.benchmark = True
    
    # Increase batch size and learning rate
    model = VOLT_A1(num_classes=10, padding=0, dim=1)
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    trainer = Trainer(Datasets("datasets", 256), model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="results")

    trainer.train(num_epoch=5)
    trainer.plot()


if __name__ == "__main__":
    train()
