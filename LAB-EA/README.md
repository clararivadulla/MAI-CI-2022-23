# Training Neural Networks with Evolutionary Algorithms 
### Benjamí Parellada & Clara Rivadulla
---
## Introduction
In this project, we've implemented the necessary files to train a neural network with the use of Evolutionary Algorithms. The weights of the NN can be trained derivative based, with a genetic algorithm, or with CMA-ES. Both genetic algorithms and CMA-ES are also used to find the best architecture.

The data sets used for training, validation and testing are generated synthetically and are related to the performance of students in the Bachelor Degree in Informatics Engineering of the UPC. All of the variables are generated randomly, but some of them are correlated to one another.

## Requirements
The imports needed to run the project are:

- `library(knitr)`
- `library(GA)`
- `library(tictoc)`
- `library(nnet)`
- `library(MASS)`
- `library(cmaesr)`

These packages might need to be installed by running `install.packages("GA")` ("GA" is just an example, simply replace it by the name of the package that needs to be installed). 

## Structure of the project

The project contains the following files:
- `NNs_EA.Rmd`, an R Markdown file used somehow as a main, where all the important functions from other files are called and results are displayed. 
- `generate_data.R`, where we define a function to **generate a synthetic dataset** with given size, data split, hardness and noisiness.
- `nnet_derivative.R`, which contains the encodings to find the best architecture using **genetic algorithms** and train the model with **backpropagation**. 
- `nnet_genetic.R`, which contains the encodings to both find the best architecture and train the neural network using **genetic algorithms**.
- `nnet_evolutionary.R`, which contains the encodings to both find the best architecture and train the neural network using **evolution strategies**.