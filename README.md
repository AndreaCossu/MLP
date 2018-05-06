# MLP
MultiLayer Perceptron implementation in Matlab. 

MLP.m is the main class and provides the implementation of the Neural Network.

ActivationFunctions/ provides some of the most popular activation functions.

ErrorFunctions/ provides some of the typical error functions for classification and regression tasks.

Utility/ provides some additional functions such as the ArmijoWolfe.m file, that implements Strong Wolfe line search procedure.

You can instantiate the main class and play with the hyperparameters to run the model on a classification problem or a regression problem. You can use L2 regularization, conjugate gradient with HS+,PR+,FR betas or gradient descent with momentum. You can optionally activate a Strong Wolfe line search for both training algorithms.
