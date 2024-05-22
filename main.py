import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import linalg as LA
import torch.optim as optim
from torchvision import datasets, transforms
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np
from src import NeuralNet, SmoothSailing, kappa

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
    model_0 = NeuralNet(size=config.layer_size).to(device)
    sail_0 = SmoothSailing(beta=0)
    optimizer_0 = optim.Adam(model_0.parameters(), lr=config.lr)

    torch.manual_seed(config.seed)
    model_0001 = NeuralNet(size=config.layer_size).to(device)
    sail_0001 = SmoothSailing(beta=0.001)
    optimizer_0001 = optim.Adam(model_0001.parameters(), lr=config.lr)

    torch.manual_seed(config.seed)
    model_001 = NeuralNet(size=config.layer_size).to(device)
    sail_001 = SmoothSailing(beta=0.01)
    optimizer_001 = optim.Adam(model_001.parameters(), lr=config.lr)

    torch.manual_seed(config.seed)
    model_01 = NeuralNet(size=config.layer_size).to(device)
    sail_01 = SmoothSailing(beta=0.1)
    optimizer_01 = optim.Adam(model_01.parameters(), lr=config.lr)

    torch.manual_seed(config.seed)
    model_1 = NeuralNet(size=config.layer_size).to(device)
    sail_1 = SmoothSailing(beta=1)
    optimizer_1 = optim.Adam(model_1.parameters(), lr=config.lr)






    fit_0 = []
    fit_0_val = []
    fit_0001 = []
    fit_0001_val = []
    fit_001 = []
    fit_001_val = []
    fit_01 = []
    fit_01_val = []
    fit_1 = []
    fit_1_val = []
    acc_0 = []
    acc_0001 = []
    acc_001 = []
    acc_01 = []
    acc_1 = []
    cond_0 = []
    cond_0001 = []
    cond_001 = []
    cond_01 = []
    cond_1 = []

    W_0 = model_0.linear1.weight.data
    cond_0.append(kappa(W_0))

    W_0001 = model_0001.linear1.weight.data
    cond_0001.append(kappa(W_0001))

    W_001 = model_001.linear1.weight.data
    cond_001.append(kappa(W_001))

    W_01 = model_01.linear1.weight.data
    cond_01.append(kappa(W_01))

    W_1 = model_1.linear1.weight.data
    cond_1.append(kappa(W_1))

    print(f"Init condition numbers:")
    print(f"\tBeta = 0: {cond_0[-1]:.2f}")
    print(f"\tBeta = 0.001: {cond_0001[-1]:.2f}")
    print(f"\tBeta = 0.01: {cond_001[-1]:.2f}")
    print(f"\tBeta = 0.1: {cond_01[-1]:.2f}")
    print(f"\tBeta = 1: {cond_1[-1]:.2f}")




    # train and evaluate

    for epoch in range(config.epochs):
        running_loss_0 = 0.0
        running_loss_0001 = 0.0
        running_loss_001 = 0.0
        running_loss_01 = 0.0
        running_loss_1 = 0.0

        running_val_loss_0 = 0.0
        running_val_loss_0001 = 0.0
        running_val_loss_001 = 0.0
        running_val_loss_01 = 0.0
        running_val_loss_1 = 0.0

        model_0.train()
        model_0001.train()
        model_001.train()
        model_01.train()
        model_1.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data.view(-1, 28*28))
            data, target = data.to(device), target.to(device)

            output_0 = model_0(data)
            loss_0 = sail_0(output_0, target)
            optimizer.zero_grad()
            loss_0.backward()
            optimizer_0.step()

            W_0001 = model_0001.linear1.weight
            output_0001 = model_0001(data)
            loss_bas_0001, loss_0001 = sail0001(output_0001, target, W_0001)
            optimizer_0001.zero_grad()
            loss_0001.backward()
            optimizer_0001.step()

            W_001 = model_001.linear1.weight
            output_001 = model_001(data)
            loss_bas_001, loss_001 = sail001(output_001, target, W_001)
            optimizer_001.zero_grad()
            loss_001.backward()
            optimizer_001.step()

            W_01 = model_01.linear1.weight
            output_01 = model_01(data)
            loss_bas_01, loss_01 = sail01(output_01, target, W_01)
            optimizer_01.zero_grad()
            loss_01.backward()
            optimizer_01.step()

            W_1 = model_1.linear1.weight
            output_1 = model_1(data)
            loss_bas_1, loss_1 = sail1(output_1, target, W_1)
            optimizer_1.zero_grad()
            loss_1.backward()
            optimizer_1.step()

            running_loss_0 += loss_0.item()
            running_loss_0001 += loss_0001.item()
            running_loss_001 += loss_001.item()
            running_loss_01 += loss_01.item()
            running_loss_1 += loss_1.item()

            if batch_idx % config.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss_0001: {:.6f},\tLoss_001: {:.6f},\tLoss_01: {:.6f},\tLoss_1: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_0.item(), loss_0001.item(), loss_001.item(), loss_01.item(), loss_1.item()))


        model_0.eval()
        model_0001.eval()
        model_001.eval()
        model_01.eval()
        model_1.eval()

        with torch.no_grad():
            for data, target in test_loader:
                data = Variable(data.view(-1, 28*28))
                data, target = data.to(device), target.to(device)

                output_0 = model_0(data)
                loss_0 = sail_0(output_0, target)

                W_0001 = model_0001.linear1.weight
                output_0001 = model_0001(data)
                loss_0001, loss_0001 = smoothsail(output_0001, target, W_0001)

                W_001 = model_001.linear1.weight
                output_001 = model_001(data)
                loss_001, loss_001 = smoothsail(output_001, target, W_001)

                W_01 = model_01.linear1.weight
                output_01 = model_01(data)
                loss_01, loss_01 = smoothsail(output_01, target, W_01)

                W_1 = model_1.linear1.weight
                output_1 = model_1(data)
                loss_1, loss_1 = smoothsail(output_1, target, W_1)

                running_val_loss_0 += loss_0.item()
                running_val_loss_0001 += loss_0001.item()
                running_val_loss_001 += loss_001.item()
                running_val_loss_01 += loss_01.item()
                running_val_loss_1 += loss_1.item()

        W_0 = model_0.linear1.weight.data
        cond_0.append(kappa(W_0))
        W_0001 = model_0001.linear1.weight.data
        cond_0001.append(kappa(W_0001))
        W_001 = model_001.linear1.weight.data
        cond_001.append(kappa(W_001))
        W_01 = model_01.linear1.weight.data
        cond_01.append(kappa(W_01))
        W_1 = model_1.linear1.weight.data
        cond_1.append(kappa(W_1))

        fit_0.append(running_loss_0/len(train_loader.dataset))
        fit_0_val.append(running_val_loss_0/config.test_batch_size)
        fit_0001.append(running_loss_0001/len(train_loader.dataset))
        fit_0001_val.append(running_val_loss_0001/config.test_batch_size)
        fit_001.append(running_loss_001/len(train_loader.dataset))
        fit_001_val.append(running_val_loss_001/config.test_batch_size)
        fit_01.append(running_loss_01/len(train_loader.dataset))
        fit_01_val.append(running_val_loss_01/config.test_batch_size)
        fit_1.append(running_loss_1/len(train_loader.dataset))
        fit_1_val.append(running_val_loss_1/config.test_batch_size)


        print(f"Epoch {epoch+1}/{config.epochs}:")
        print(f"\tBeta = 0, Loss: {fit_0[-1]:.2f} with condition number {cond_0[-1]:.2f}")
        print(f"\tBeta = 0.001, Loss: {fit_0001[-1]:.2f} with condition number {cond_0001[-1]:.2f}")
        print(f"\tBeta = 0.01, Loss: {fit_001[-1]:.2f} with condition number {cond_001[-1]:.2f}")
        print(f"\tBeta = 0.1, Loss: {fit_01[-1]:.2f} with condition number {cond_01[-1]:.2f}")
        print(f"\tBeta = 1, Loss: {fit_1[-1]:.2f} with condition number {cond_1[-1]:.2f}")


    # save results as dictionary
    results = {
        "fit_0": fit_0,
        "fit_0_val": fit_0_val,
        "fit_0001": fit_0001,
        "fit_0001_val": fit_0001_val,
        "fit_001": fit_001,
        "fit_001_val": fit_001_val,
        "fit_01": fit_01,
        "fit_01_val": fit_01_val,
        "fit_1": fit_1,
        "fit_1_val": fit_1_val,
        "cond_0": cond_0,
        "cond_0001": cond_0001,
        "cond_001": cond_001,
        "cond_01": cond_01,
        "cond_1": cond_1,
    }

    with open(f"results_{config.beta}.json", "w") as f:
        json.dump(results, f)
    
    # save models
    torch.save(model_0.state_dict(), "model_0_new.pt")
    torch.save(model_0001.state_dict(), "model_0001_new.pt")
    torch.save(model_001.state_dict(), "model_001_new.pt")
    torch.save(model_01.state_dict(), "model_01_new.pt")
    torch.save(model_1.state_dict(), "model_1_new.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs (default: 100)"
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