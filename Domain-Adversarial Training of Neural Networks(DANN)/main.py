import torch
import torch.nn as nn
import dataloader
import model
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

batch_size = 500
epoch_size = 100
learning_rate = 0.01
beta = 0.6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def optimizer_update(optimizer, p):
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75
    return optimizer

def print_img(img, target):
    img = img.cpu()
    img = img * 0.5
    img = img + 0.5
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.title(str(target.item()))
    plt.show()

S_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/chentianle/data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=batch_size, shuffle=False)

T_loader = dataloader.load_train_data('/home/chentianle/data/MNIST_M/train', batch_size=batch_size)
Test_loader = dataloader.load_test_data('/home/chentianle/data/MNIST_M/test', batch_size=batch_size)


feature_extractor = model.Feature_Extractor().to(device)
class_classifier = model.Class_classifier().to(device)
domain_classifier = model.Domain_classifier().to(device)

optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                        {'params': class_classifier.parameters()},
                        {'params': domain_classifier.parameters()}], lr=learning_rate, momentum= 0.9)

class_criterion = nn.NLLLoss().to(device)
domain_criterion = nn.NLLLoss().to(device)
maxn = 0.72

feature_extractor.load_state_dict(torch.load("./feature_extractor.pth"))
class_classifier.load_state_dict(torch.load("./class_classifier.pth"))
domain_classifier.load_state_dict(torch.load("./domain_classifier.pth"))

for epoch in range(epoch_size):

    iter_source = iter(S_loader)
    iter_target = iter(T_loader)
    len_source = len(iter_source)
    len_target = len(iter_target)

    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()

    for batch_id in range(len_source):

        data_S, label_S = iter_source.next()
        data_T, _ = iter_target.next()
        # print_img(data_T[0], _[0])

        data_S = data_S.repeat((1, 3, 1, 1))
        data_S = data_S.to(device)
        label_S = label_S.to(device)
        data_T = data_T.to(device)

        p = 0.5 * float(batch_id + epoch * len_source) / (len_source * epoch_size)

        optimizer = optimizer_update(optimizer, p)
        optimizer.zero_grad()

        beta = 2. / (1. + np.exp(-10 * p)) - 1

        if (batch_id+1) % len_target == 0:
            iter_target = iter(T_loader)
        if (batch_id+1) % len_source == 0:
            iter_source = iter(S_loader)

        feature_source = feature_extractor(data_S)
        feature_target = feature_extractor(data_T)

        class_pred = class_classifier(feature_source)
        class_loss = class_criterion(class_pred, label_S)

        domain_pred_S = domain_classifier(feature_source, beta)
        domain_pred_T = domain_classifier(feature_target, beta)

        source_labels = torch.zeros((domain_pred_S.shape[0])).type(torch.LongTensor).to(device)
        target_labels = torch.ones((domain_pred_T.shape[0])).type(torch.LongTensor).to(device)

        domain_loss = domain_criterion(domain_pred_S, source_labels) + domain_criterion(domain_pred_T, target_labels)

        loss = class_loss + domain_loss
        loss.backward()
        optimizer.step()

        if (batch_id + 1) % 10 == 0:
            print('Epoch: {}[{}/{} ({:.0f}%)]\nLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                epoch, batch_id * batch_size, len(S_loader.dataset),
                100. * batch_id / len(S_loader), loss.item(), class_loss.item(),
                domain_loss.item()
            ))

    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()

    test_loss = 0
    correct = 0
    for i, (data, target) in enumerate(S_loader):

        data = data.repeat((1, 3, 1, 1))

        data = data.to(device)
        target = target.to(device)

        features = feature_extractor(data)
        class_pred = class_classifier(features)
        test_loss += class_criterion(class_pred, target).item()

        pred = class_pred.argmax(dim=1)
        # print(pred)
        # print(target)
        correct += pred.eq(target).float().sum().item()

    test_loss /= len(S_loader)
    print('Source set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(S_loader.dataset),
        100. * correct / len(S_loader.dataset)))

    test_loss = 0
    correct = 0
    for i, (data, target) in enumerate(Test_loader):
        # print_img(data[0], target[0])

        data = data.to(device)
        target = target.to(device)

        features = feature_extractor(data)
        class_pred = class_classifier(features)
        test_loss += class_criterion(class_pred, target).item()

        pred = class_pred.argmax(dim=1)
        # print(pred)
        # print(target)
        correct += pred.eq(target).float().sum().item()

    test_loss /= len(Test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(Test_loader.dataset),
        100. * correct / len(Test_loader.dataset)))


    if correct / len(Test_loader.dataset) > maxn:
        torch.save(obj=feature_extractor.state_dict(), f="./feature_extractor.pth")
        torch.save(obj=class_classifier.state_dict(), f="./class_classifier.pth")
        torch.save(obj=domain_classifier.state_dict(), f="./domain_classifier.pth")
        maxn = correct / len(T_loader.dataset)