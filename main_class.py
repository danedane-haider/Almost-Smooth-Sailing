import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from types import SimpleNamespace
import numpy as np
from models import NeuralNet
from loss import SmoothSailing, kappa

def main(args):
    config = SimpleNamespace(batch_size=32,
                            test_batch_size=1000,
                            epochs=args.epochs,
                            lr=args.lr,
                            momentum=0.5,
                            seed=1,
                            log_interval=100,
                            beta=args.beta,
                            layer_size=args.layer_size)
    torch.manual_seed(config.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    print('lr: ', config.lr)
    print('beta: ', config.beta)
    print('layer_size: ', config.layer_size)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
        batch_size=config.batch_size, shuffle=True, **kwargs)
        
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=config.test_batch_size, shuffle=True, **kwargs)




    torch.manual_seed(config.seed)
    model = NeuralNet(size=config.layer_size).to(device)
    sail = SmoothSailing(beta=config.beta)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)


    fit = []
    fit_val = []
    cond = []

    W = model.linear1.weight.data
    cond.append(kappa(W))

    print(f"Init condition number: {cond[-1]:.2f}")



    # train and evaluate

    for epoch in range(config.epochs):
        running_loss = 0.0
        running_val_loss = 0.0

        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data.view(-1, 28*28))
            data, target = data.to(device), target.to(device)

            W = model.linear1.weight
            output = model(data)
            loss_bas, loss = sail(output, target, W)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % config.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_bas.item()))


        model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                data = Variable(data.view(-1, 28*28))
                data, target = data.to(device), target.to(device)

                W = model.linear1.weight
                output_0 = model(data)
                loss_bas_val, loss_val = sail(output, target, W)

                running_val_loss += loss_bas_val.item()

        W = model.linear1.weight.data
        cond.append(kappa(W))

        fit.append(running_loss/len(train_loader.dataset))
        fit_val.append(running_val_loss/config.test_batch_size)

        print(f"Epoch {epoch+1}/{config.epochs}:")
        print(f"\tLoss: {fit[-1]:.2f} with condition number {cond[-1]:.2f}")


    # save models
    torch.save(model.state_dict(), "model.pt")

    results = {
        "fit": fit,
        "fit_val": fit_val,
        "cond": cond,
    }

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs (default: 100)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1,
        help="Beta (default:  1)",
    )
    parser.add_argument(
        "--layer_size",
        type=int,
        default=2048,
        help="Size of the hidden layer (default: 2048)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate of the optimizer.",
    )

    main(parser.parse_args())