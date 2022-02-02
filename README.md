# function-approximation-nn
A project in collaboration with Darin K. and Ian A.
It is based around [this article](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6).
## Description
---
This project is a model of a neural network built without machine learning libraries, such as Pytorch or TensorFlow. It focuses on using other scientific Python packages (such as NumPy) to implement machine learning algorithms and present them as mathematical expressions. The neural network predicts values of a given function for arguments which it hasn't 'seen' yet after being trained on test data of x and pre-calculated y values. At the end, a plot is shown with a graphical representation of the user-defined function with the guessed values marked with red dots. It depends on the binary representation of integers (both positive and negative) and contains an input, two hidden and an output layer.

## Usage
---
The following macros set the number of neurons in the hidden layers
```python 
neus1 = 16  # neurons per layer
neus2 = 16
```
The next two variables set the number of neurons in the input and output layers. Since every node in these layers is either 0 or 1, as in the binary description of a decimal number, these act also as encoding of the input and output.
```python 
bitIn = 12  # encoding of input
bitOut = 16  # encoding of output
```
In line 12 the range of x-values, which later will be fed into a function and into our model, is described.

```python 
y = np.array(np.int64(19 * np.sin(X / 47) - 32 * np.cos(X / 100)))
```
This line sets the function that a user will want the network to guess, or 'approximate' (note that the graphic of the function drawn by pyplot must also be adjusted accordingly. It can be found under #draws plot.)

One more thing to keep in mind is the adjustment of the training loops.
