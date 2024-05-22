import argparse
import pickle
import json
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
from src import AutoEncoder, SmoothSailingAE, kappa

def main(args):
    config = SimpleNamespace(batch_size=32,
                            test_batch_size=1000,
                            epochs=args.epochs,
                            lr=args.lr,
                            momentum=0.5,
                            seed=1,
                            log_interval=100,
                            beta_end=args.beta_end,
                            beta_mid=args.beta_mid,
                            end_layer_size=args.end_layer_size,
                            mid_layer_size=args.mid_layer_size,
                            noise_level=args.noise_level)
    torch.manual_seed(config.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    print('lr: ', config.lr)
    print('beta_end: ', config.beta_end)
    print('beta_mid: ', config.beta_mid)
    print('end_layer_size: ', config.end_layer_size)
    print('mid_layer_size: ', config.mid_layer_size)
    print('noise_level: ', config.noise_level)

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
    model = AutoEncoder(end_layer_size=config.end_layer_size, mid_layer_size=config.mid_layer_size).to(device)
    sail = SmoothSailingAE(beta_end=0)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    torch.manual_seed(config.seed)
    model_reg = AutoEncoder(end_layer_size=config.end_layer_size, mid_layer_size=config.mid_layer_size).to(device)
    smoothsail = SmoothSailingAE(beta_end=config.beta_end, beta_mid=config.beta_mid)
    optimizer_reg = optim.Adam(model_reg.parameters(), lr=config.lr)





    fit = []
    fit_val = []
    fit_reg = []
    fit_reg_val = []

    cond_enc = []
    cond_mid1 = []
    cond_mid2 = []
    cond_dec = []
    cond_enc_reg = []
    cond_mid1_reg = []
    cond_mid2_reg = []
    cond_dec_reg = []

    W1 = model.linear1.weight.data
    M1 = model.middle1.weight.data
    M2 = model.middle2.weight.data
    W2 = model.linear2.weight.data
    cond_enc.append(kappa(W1))
    cond_mid1.append(kappa(M1))
    cond_mid2.append(kappa(M2))
    cond_dec.append(kappa(W2))

    W1_reg = model_reg.linear1.weight.data
    M1_reg = model_reg.middle1.weight.data
    M2_reg = model_reg.middle2.weight.data
    W2_reg = model_reg.linear2.weight.data
    cond_enc_reg.append(kappa(W1_reg))
    cond_mid1_reg.append(kappa(M1_reg))
    cond_mid2_reg.append(kappa(M2_reg))
    cond_dec_reg.append(kappa(W2_reg))

    print(f"Init condition numbers:")
    print(f"\tCondition numbers {cond_enc[-1]:.2f}", f"{cond_mid1[-1]:.2f}", f"{cond_mid2[-1]:.2f}", f"{cond_dec[-1]:.2f}")
    print(f"\tReguarized condition numbers {cond_enc_reg[-1]:.2f}", f"{cond_mid1_reg[-1]:.2f}", f"{cond_mid2_reg[-1]:.2f}", f"{cond_dec_reg[-1]:.2f}")






    # train and evaluate

    for epoch in range(config.epochs):
        running_loss = 0.0
        running_loss_reg = 0.0

        running_val_loss = 0.0
        running_val_loss_reg = 0.0

        model.train()
        model_reg.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            noisy_data = data + args.noise_level * torch.randn(data.shape)
            data = Variable(data.view(-1, 28*28))
            target = Variable(noisy_data.view(-1, 28*28))
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = sail(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            W1 = model_reg.linear1.weight
            M1 = model_reg.middle1.weight
            M2 = model_reg.middle2.weight
            W2 = model_reg.linear2.weight
            
            output_reg = model_reg(data)
            loss_bas, loss_reg = smoothsail(output_reg, target, W1, M1, M2, W2)
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
                noisy_data = data + args.noise_level * torch.randn(data.shape)
                data = Variable(data.view(-1, 28*28))
                target = Variable(noisy_data.view(-1, 28*28))
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = sail(output, target)

                W1 = model_reg.linear1.weight
                M1 = model_reg.middle1.weight
                M2 = model_reg.middle2.weight
                W2 = model_reg.linear2.weight

                output_reg = model_reg(data)
                loss_bas, loss_reg = smoothsail(output_reg, target, W1, M1, M2, W2)

                running_val_loss += loss.item()
                running_val_loss_reg += loss_bas.item()

        W1 = model.linear1.weight.data
        M1 = model.middle1.weight.data
        M2 = model.middle2.weight.data
        W2 = model.linear2.weight.data
        cond_enc.append(kappa(W1))
        cond_mid1.append(kappa(M1))
        cond_mid2.append(kappa(M2))
        cond_dec.append(kappa(W2))

        W1_reg = model_reg.linear1.weight.data
        M1_reg = model_reg.middle1.weight.data
        M2_reg = model_reg.middle2.weight.data
        W2_reg = model_reg.linear2.weight.data
        cond_enc_reg.append(kappa(W1_reg))
        cond_mid1_reg.append(kappa(M1_reg))
        cond_mid2_reg.append(kappa(M2_reg))
        cond_dec_reg.append(kappa(W2_reg))


        fit.append(running_loss/len(train_loader.dataset))
        fit_val.append(running_val_loss/config.test_batch_size)
        fit_reg.append(running_loss_reg/len(train_loader.dataset))
        fit_reg_val.append(running_val_loss_reg/config.test_batch_size)


        print(f"Epoch {epoch+1}/{config.epochs}:")
        print(f"\tLoss: {fit[-1]:.2f} with condition numbers {cond_enc[-1]:.2f}", f"{cond_mid1[-1]:.2f}", f"{cond_mid2[-1]:.2f}", f"{cond_dec[-1]:.2f}")
        print(f"\tRegularized Loss: {fit_reg[-1]:.2f} with condition numbers {cond_enc_reg[-1]:.2f}", f"{cond_mid1_reg[-1]:.2f}", f"{cond_mid2_reg[-1]:.2f}", f"{cond_dec_reg[-1]:.2f}")


    # save models
    torch.save(model.state_dict(), f"denoise_model{args.noise_level}.pt")
    torch.save(model_reg.state_dict(), f"denoise_model_reg{args.noise_level}.pt")

    # save results detached from the gradient graph
    denoise_results = {
        "fit": fit,
        "fit_val": fit_val,
        "fit_reg": fit_reg,
        "fit_reg_val": fit_reg_val,
        "cond_enc": cond_enc,
        "cond_mid1": cond_mid1,
        "cond_mid2": cond_mid2,
        "cond_dec": cond_dec,
        "cond_enc_reg": cond_enc_reg,
        "cond_mid1_reg": cond_mid1_reg,
        "cond_mid2_reg": cond_mid2_reg,
        "cond_dec_reg": cond_dec_reg,
    }

    with open(f"denoise_results{args.noise_level}.pkl", "wb") as f:
        pickle.dump(denoise_results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs (default: 100)"
    )
    parser.add_argument(
        "--beta_end",
        type=float,
        default=0.05,
        help="Beta for encoder/decoder (default:  1)",
    )
    parser.add_argument(
        "--beta_mid",
        type=float,
        default=0.01,
        help="Beta for middle layers (default: 1)",
    )
    parser.add_argument(
        "--end_layer_size",
        type=int,
        default=256,
        help="Size of the hidden layer (default: 256)",
    )
    parser.add_argument(
        "--mid_layer_size",
        type=int,
        default=32,
        help="Size of the hidden layer (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate of the optimizer.",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.1,
        help="Noise level for the data.",
    )


    main(parser.parse_args())