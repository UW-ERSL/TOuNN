# Fourier-TOuNN

[Length Scale Control in Topology Optimization using Fourier Enhanced Neural Networks](https://ersl.wisc.edu/publications/2020/FourierTOuNN.pdf)

Aaditya Chandrasekhar, Krishnan Suresh  
[Engineering Representations and Simulation Lab](https://ersl.wisc.edu)  
University of Wisconsin-Madison 

## Abstract
Length scale control is often imposed in topology optimization (TO) to make the design amenable to manufacturing and other functional requirements. While several length scale control strategies have been proposed, practical challenges often arise due to the coupling between the mesh size and length scales. In particular, when a maximum length scale is imposed to obtain thin members, extraction of the boundary is often impaired by the mesh resolution.

In this paper we propose a mesh-independent length scale control strategy, by extending a recently proposed SIMP-based TO formulation using neural networks (TOuNN). Specifically, we enhance TOuNN with a Fourier space projection, to control the minimum and/or maximum length scales. The proposed method does not involve additional constraints, while the sensitivity computations are automated through the neural network’s backpropagation, making the method easy to implement.

## Code

Run FourierTOuNN IPython notebook on Google Colab for a demo.
