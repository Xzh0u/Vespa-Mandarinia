from __future__ import print_function
import argparse
import torch
from skimage import io
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from utils.util import *
from utils.gradual_domain_adaptation import *
from data_loader.datasets import PACSDataset
from data_loader.data_loaders import PACSDataloader
import torchvision.models as models
from trainer.train import train, test


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PACS GDA Experiment')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--window-size', type=int, default=10,
                        help='window size when do gradual self training')
    parser.add_argument('--classtype', type=str, default="photo",
                        help='which class as the target')
    parser.add_argument('--retrain-base', type=bool, default=False,
                        help='whether retrain the base classifier')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}
    if torch.cuda.is_available():
        kwargs.update({'num_workers': 1,
                       'pin_memory': True},
                      )

    # load train data
    test_set, train_set, val_set = PACSDataloader(
        args.classtype)  # train: source | test: target
    train_loader = DataLoader(train_set, **kwargs)
    val_loader = DataLoader(val_set, **kwargs)
    test_loader = DataLoader(test_set, **kwargs)
    print("Finish loading PACS data.")

    # [if required] retrain base model
    if args.retrain_base:
        model = models.alexnet(pretrained=True).to(device)
        model.classifier[6] = nn.Linear(4096, 7).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        scheduler = ReduceLROnPlateau(optimizer)

        for epoch in range(1, args.epochs + 1):
            train(args.log_interval, model, device,
                  train_loader, optimizer, epoch)
            val_loss = test(model, device, test_loader)
            scheduler.step(val_loss)

        torch.save(model.state_dict(),
                   f"saved/models/PACS/alex_base_{args.classtype}.pt")
        print("Model saved.")

    # load weights and do experiments
    num_classes = 7
    model = models.alexnet(pretrained=True).to(device)
    model.classifier[6] = nn.Linear(4096, num_classes).to(device)
    model.load_state_dict(torch.load(
        f"saved/models/PACS/alex_base_{args.classtype}.pt"))
    model.eval()
    print("Finish loading weights.")

    # directly test on photos
    print("##### Directly test on photos #####")
    test(model, device, test_loader)


if __name__ == '__main__':
    main()
