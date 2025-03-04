from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.util import *
from data_loader.data_loader import HornetDataset, HornetTestDataset
import torchvision.models as models
from trainer.train import train, test, predict
# from torch.utils.tensorboard import SummaryWriter
import torchvision

# writer = SummaryWriter("./saved/log")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--retrain-base', type=bool, default=True,
                        help='whether retrain the base classifier')
    parser.add_argument('--log-interval', type=int, default=4, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}
    if torch.cuda.is_available():
        kwargs.update({'num_workers': 1,
                       'pin_memory': True},
                      )

    # load train/test data
    train_set = HornetDataset()
    test_set = HornetTestDataset()
    train_loader = DataLoader(train_set, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False)
    print("Finish loading training data.")
    # imgshow(test_loader)
    # train base model
    if args.retrain_base:
        model = models.alexnet(pretrained=True).to(device)
        model.classifier[6] = nn.Linear(4096, 2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        # scheduler = ReduceLROnPlateau(optimizer)

        for epoch in range(1, args.epochs + 1):
            train(args.log_interval, model, device,
                  train_loader, optimizer, epoch)
            # val_loss = test(model, device, test_loader)
            # scheduler.step(val_loss)

        torch.save(model.state_dict(), "saved/model/alex_hornet.pt")
        print("Model saved.")

    # load weights and do experiments
    model = models.alexnet(pretrained=True).to(device)
    model.classifier[6] = nn.Linear(4096, 2).to(device)
    model.load_state_dict(torch.load("saved/model/alex_hornet.pt"))
    model.eval()
    print("Finish loading weights.")

    # directly test on photos
    print("Directly test on photos")
    predict(model, device, test_loader, test_set)


if __name__ == '__main__':
    main()
