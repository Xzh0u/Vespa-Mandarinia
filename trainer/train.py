import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    running_corrects = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.float())
        criterian = nn.CrossEntropyLoss()
        loss = criterian(output, target.flatten().type(torch.cuda.LongTensor))
        _, preds = torch.max(output, dim=1)
        loss.backward()
        optimizer.step()
        running_corrects += torch.sum(preds == target.data)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), epoch_acc))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterian = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            # sum up batch loss
            test_loss += criterian(output,
                                   target.flatten().type(torch.cuda.LongTensor)).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss
