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
    model = NeuralNet(size=config.layer_size).to(device)
    sail = SmoothSailing(beta=0)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    torch.manual_seed(config.seed)
    model_reg = NeuralNet(size=config.layer_size).to(device)
    smoothsail = SmoothSailing(beta=config.beta)
    optimizer_reg = optim.Adam(model_reg.parameters(), lr=config.lr)





    fit = []
    fit_val = []
    fit_reg = []
    fit_reg_val = []
    acc = []
    acc_reg = []
    cond = []
    cond_reg = []

    W = model.linear1.weight.data
    cond.append(kappa(W))

    W_reg = model_reg.linear1.weight.data
    cond_reg.append(kappa(W_reg))

    print(f"Init condition numbers:")
    print(f"\tBaseline condition number {cond[-1]:.2f}")
    print(f"\tReguarized condition number {cond_reg[-1]:.2f}")






    # train and evaluate

    for epoch in range(config.epochs):
        running_loss = 0.0
        running_loss_reg = 0.0

        running_val_loss = 0.0
        running_val_loss_reg = 0.0

        model.train()
        model_reg.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data.view(-1, 28*28))
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = sail(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_reg = model_reg(data)
            W = model_reg.linear1.weight
            loss_bas, loss_reg = smoothsail(output_reg, target, W)
            optimizer_reg.zero_grad()
            loss_reg.backward()
            optimizer_reg.step()

            running_loss += loss.item()
            running_loss_reg += loss_bas.item()

            if batch_idx % config.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss_reg: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), loss_bas.item()))


        model.eval()
        model_reg.eval()

        with torch.no_grad():
            for data, target in test_loader:
                data = Variable(data.view(-1, 28*28))
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = sail(output, target)

                output_reg = model_reg(data)
                W = model_reg.linear1.weight
                loss_bas, loss_reg = smoothsail(output_reg, target, W)

                running_val_loss += loss.item()
                running_val_loss_reg += loss_bas.item()

                pred = output.max(1, keepdim=True)[1]
                pred_reg = output_reg.max(1, keepdim=True)[1]
                accur = pred.eq(target.view_as(pred)).sum().item()
                accur_reg = pred_reg.eq(target.view_as(pred_reg)).sum().item()


        W = model.linear1.weight.data
        cond.append(kappa(W))
        W_reg = model_reg.linear1.weight.data
        cond_reg.append(kappa(W_reg))

        fit.append(running_loss/len(test_loader.dataset))
        fit_val.append(running_val_loss/config.test_batch_size)
        fit_reg.append(running_loss_reg/len(test_loader.dataset))
        fit_reg_val.append(running_val_loss_reg/config.test_batch_size)

        acc.append(accur/config.test_batch_size*100)
        acc_reg.append(accur_reg/config.test_batch_size*100)


        print(f"Epoch {epoch+1}/{config.epochs}:")
        print(f"\tLoss: {fit[-1]:.2f} with condition number {cond[-1]:.2f}")
        print(f"\tRegularized Loss: {fit_reg[-1]:.2f} with condition number {cond_reg[-1]:.2f}")
        print(f"\tAccuracy: {acc[-1]:.2f}%")
        print(f"\tRegularized Accuracy: {acc_reg[-1]:.2f}%")


    # save results
    results = {
        "fit": fit,
        "fit_val": fit_val,
        "fit_reg": fit_reg,
        "fit_reg_val": fit_reg_val,
        "acc": acc,
        "acc_reg": acc_reg,
        "cond": cond,
        "cond_reg": cond_reg
    }

    with open(f"results_{config.beta}.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # save models
    torch.save(model.state_dict(), f"model_{config.beta}.pt")
    torch.save(model_reg.state_dict(), f"model_reg_{config.beta}.pt")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=51, help="Number of epochs (default: 51)"
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