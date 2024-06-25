# (Almost) Smooth Sailing:
## Towards Numerical Stability of Neural Networks Through Differentiable Regularization of the Condition Number

![MNIST_denoising](https://github.com/danedane-haider/Almost-Smooth-Sailing/assets/55834940/cebbbfac-0099-402d-8d57-3102aef4f02e)

## About

Maintaining numerical stability in networks is crucial for their reliability and performance. One approach to maintain stability of a network layer is to integrate the condition number of the weight matrix as a regularizing term into the optimization algorithm. However, due to its discontinuous nature and lack of differentiability the condition number is not suitable for a gradient descent approach. We introduces a novel regularizer that is provably differentiable almost everywhere and promotes matrices with low condition numbers.

Let $S\in\mathbb{R}^{n\times m}$ be a matrix and let $\nu=\min(n,m).$ The condition number of $S$ is defined as
$$\kappa (S):={\Vert S\Vert}_2{\Vert S^\dagger\Vert}_2,$$
where $S^\dagger$ is the pseudo inverse of $S$. The proposed regularizer is given by
$$r(S) := \frac{1}{2}\Vert S\Vert_2^2-\frac{1}{2\nu}\Vert S\Vert_F^2,$$
where $\Vert\cdot\Vert_2$ denotes the spectral norm and $\Vert\cdot\Vert_F$ the Frobenius-norm. We can prove that

$$r(S) = 0$ \text{ if and only if } $S$ \text{ has full rank and } $\kappa(S)=1.$$

This repository provides all functions that are needed to use the proposed regularizer in any desired setting.

## Citation

Please cite

```
@inproceedings{Nenov2024smoothsailing,
  title = {({A}lmost) {S}mooth {S}ailing: Towards Numerical Stability of Neural Networks Through Differentiable Regularization of the Condition Number},
  author = {Nenov, Rossen and Haider, Daniel and Balazs, Peter},
  booktitle = {2nd Differentiable Almost Everything Workshop at the 41st International Conference on Machine Learning},
  year = {2024}
}
```
