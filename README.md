# TOuNN

[TOuNN: Topology Optimization using Neural Networks](https://link.springer.com/article/10.1007/s00158-020-02748-4)

[Aaditya Chandrasekhar](https://aadityacs.github.io/), Krishnan Suresh  
[Engineering Representations and Simulation Lab](https://ersl.wisc.edu)  
University of Wisconsin-Madison 

## Abstract
Neural networks, and more broadly, machine learning techniques, have been recently exploited to accelerate topology optimization through data-driven training and image processing. In this paper, we demonstrate that one can directly execute topology optimization (TO) using neural networks (NN). The primary concept is to use the NN’s activation functions to represent the popular Solid Isotropic Material with Penalization (SIMP) density field. In other words, the density function is parameterized by the weights and bias associated with the NN, and spanned by NN’s activation functions; the density representation is thus independent of the finite element mesh. Then, by relying on the NN’s built-in backpropogation, and a conventional finite element solver, the density field is optimized. Methods to impose design and manufacturing constraints within the proposed framework are described and illustrated. A byproduct of representing the density field via activation functions is that it leads to a crisp and differentiable boundary. The proposed framework is simple to implement and is illustrated through 2D and 3D examples. Some of the unresolved challenges with the proposed framework are also summarized.

## Code

Run TOuNN IPython notebook on Google Colab for a demo.
