import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class Feature_Extractor(nn.Module):
    def __init__(self):
        super(Feature_Extractor, self).__init__()
        # self.features = nn.Sequential( #28*28
        #     nn.Conv2d(3, 64, kernel_size=5),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 64, kernel_size=5),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        return x


class Class_classifier(nn.Module):
    def __init__(self):
        super(Class_classifier, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(128 * 3 * 3, 3072),
        #     nn.BatchNorm1d(3072),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(3072, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(2048, 10)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return F.log_softmax(self.classifier(x), 1)

class Domain_classifier(nn.Module):
    def __init__(self):
        super(Domain_classifier, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(128 * 3 * 3, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(1024, 2)
        # )

        self.classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2)
        )

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        return F.log_softmax(self.classifier(x), 1)
