import numpy as np
import random
import matplotlib.pyplot as plt

# keyboard.press_and_release('ctrl + l')

# global variables
neus1 = 16  # neurons per layer
neus2 = 16

bitIn = 12  # encoding of input
bitOut = 16  # encoding of output
# x, y
x1 = np.array(np.linspace(-1000, 1000, 1800), dtype=np.int64)
X1 = []

for x in range(len(x1)):
    X1.append(random.choice(x1))
X = np.array(X1)
y = np.array(np.int64(19 * np.sin(X / 47) - 32 * np.cos(X / 100)))


def reLu(x):
    return x * (x > 0)


def reLuDerivative(x):
    return 1 * (x > 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x):
    return x * (1 - x)


def tanh(x):

 return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))   

def tanhDerivative(x):
    return 1 - x ** 2


af = lambda arg: tanh(arg)

af_d = lambda argg: tanhDerivative(argg)


class NeuralNetwork:
    # constructor
    def __init__(self, x, y):

        # X
        self.input1 = np.zeros((1, bitIn), dtype=int)  # [[0]

        if np.binary_repr(x)[0] == "-":
            self.input1[0][bitIn - 1] = 1
            for j in range(len(np.binary_repr(x)) - 1):
                self.input1[0][j] = int(
                    np.binary_repr(x)[len(np.binary_repr(x)) - j - 1]
                )
        else:
            for j in range(len(np.binary_repr(x))):
                self.input1[0][j] = int(
                    np.binary_repr(x)[len(np.binary_repr(x)) - j - 1]
                )

        self.input = np.array((self.input1), dtype=float)

        # weights
        self.weights1 = np.random.rand(bitIn, neus1)
        self.weights2 = np.random.rand(neus1, neus2)
        self.weights3 = np.random.rand(neus2, bitOut)

        # biases
        self.b1 = np.random.rand(1, neus1)
        self.b2 = np.random.rand(1, neus2)
        self.b3 = np.random.rand(1, bitOut)

        # y
        self.y1 = np.empty((1, bitOut), dtype=np.int64)

        if np.binary_repr(y)[0] == "-":
            self.y1[0][bitOut - 1] = 1
            for j in range(len(np.binary_repr(y)) - 1):
                self.y1[0][j] = int(np.binary_repr(y)[len(np.binary_repr(y)) - j - 1])
        else:
            for j in range(len(np.binary_repr(y))):
                self.y1[0][j] = int(np.binary_repr(y)[len(np.binary_repr(y)) - j - 1])

        self.y = np.array((self.y1), dtype=float)
        # output
        self.output = np.zeros(self.y.shape)

    def feedForward(self):

        self.layer1 = af(
            np.dot(self.input, self.weights1) + self.b1
        )  # //0.0001/1000  )//0.0001/1000
        self.layer2 = af(
            np.dot(self.layer1, self.weights2) + self.b2
        )  # //0.0001/1000)//0.0001/1000
        self.layer3 = af(np.dot(self.layer2, self.weights3) + self.b3)

        self.output = self.layer3

        return self.layer3

    def backpropagation(self):

        d_weights3 = np.dot(self.layer2.T, 2 * (self.y - self.output)*af_d(self.output))
        d_weights2 = np.dot(self.layer1.T , 2 * (self.y - self.output)* af_d(self.output)*af_d(self.layer2))
        d_weights1 = np.dot(self.input.T, 2 * (self.y - self.output)* af_d(self.output)*af_d(self.layer2)*af_d(self.layer1))

        d_b3 = 2 * (self.y - self.output) * af_d(self.output)
        d_b2 = np.dot(2 * (self.y - self.output) * af_d(self.output), self.weights3.T) * af_d(self.layer2)
        d_b1 = (np.dot(np.dot(2 * (self.y - self.output) * af_d(self.output), self.weights3.T)* af_d(self.layer2),self.weights2.T,)* af_d(self.layer1))
        # update
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3

        self.b1 += d_b1
        self.b2 += d_b2
        self.b3 += d_b3

    def train(self, x, y):
        # converts from base 10 to binary

        input1 = np.zeros((1, bitIn), dtype=int)  # [[0]

        if np.binary_repr(x)[0] == "-":
            input1[0][bitIn - 1] = 1
            for j in range(len(np.binary_repr(x)) - 1):
                input1[0][j] = int(np.binary_repr(x)[len(np.binary_repr(x)) - j - 1])
        else:
            for j in range(len(np.binary_repr(x))):
                input1[0][j] = int(np.binary_repr(x)[len(np.binary_repr(x)) - j - 1])

        self.input = np.array((input1), dtype=float)

        y1 = np.zeros((1, bitOut), dtype=np.int64)

        if np.binary_repr(y)[0] == "-":
            y1[0][bitOut - 1] = 1
            for j in range(len(np.binary_repr(y)) - 1):
                y1[0][j] = int(np.binary_repr(y)[len(np.binary_repr(y)) - j - 1])
        else:
            for j in range(len(np.binary_repr(y))):
                y1[0][j] = int(np.binary_repr(y)[len(np.binary_repr(y)) - j - 1])

        self.y = np.array((y1), dtype=float)

        # calls feedforward and backpropagation
        output = self.feedForward()
        self.backpropagation()

    def test(self, x):
        input1 = np.zeros((1, bitIn), dtype=int)  # [[0]

        if np.binary_repr(x)[0] == "-":
            input1[0][bitIn - 1] = 1
            for j in range(len(np.binary_repr(x)) - 1):
                input1[0][j] = int(np.binary_repr(x)[len(np.binary_repr(x)) - j - 1])
        else:
            for j in range(len(np.binary_repr(x))):
                input1[0][j] = int(np.binary_repr(x)[len(np.binary_repr(x)) - j - 1])

        self.input = np.array((input1), dtype=float)

        output = self.feedForward()


# x = y
loss = []
nn = NeuralNetwork(X[0], y[0])
# training loop
for u in range(100):
    for i in range(0, 1800):
        if i * u % 1000 == 0 and u != 0:
            # converts to base 10

            X10 = 0
            for j in range(bitIn - 1):
                if nn.input[0][j] == 1:
                    X10 += 2 ** j
            if nn.input[0][bitIn - 1] == 1:
                X10 = -1 * X10

            y10 = 0
            for j in range(bitOut - 1):
                if nn.y[0][j] == 1:
                    y10 += 2 ** j
            if nn.y[0][bitOut - 1] == 1:
                y10 = -1 * y10

            outp10 = 0
            for j in range(bitOut - 1):
                if nn.output[0][j] > 0.4999:
                    outp10 += 2 ** j
            if nn.output[0][bitOut - 1] > 0.4999:
                outp10 = -1 * outp10
            print("for iteration # " + str(u) + "x" + str(i) + "\n")
            print("Input : \n" + str(X10))
            print("Actual Output: \n" + str(y10))
            print("Predicted Output: \n" + str(outp10))

            # print ("Input : \n" , nn.input)
            # print ("Actual Output: \n" , nn.y)
            # print ("Predicted Output: \n" , nn.output)

            print("Loss: \n", (abs(y10 - outp10)))  # mean sum squared loss
            # loss.append(((y10 - outp10)**2)/(100/y))
            print("\n")

        nn.train(X[i], y[i])

else:
    print("weights1:", nn.weights1, "weights2:", nn.weights2, "weights3:", nn.weights3)
    print("biases1: ", nn.b1, "\nbiases2:", nn.b2, "biases3:", nn.b3)
    with open(r"nn_data.txt", "w") as f:
        f.write("[")
        for k in range(len(nn.weights1)):
            build = "["
            for l in range(len(nn.weights1[0])):
                build += str(nn.weights1[k][l])
                if l != (len(nn.weights1[0]) - 1):
                    build += ", "
            if k != (len(nn.weights1) - 1):
                build += "],\n"
            else:
                build += "]"
            f.write(build)
        f.write("]\n")
        f.write("\n")
        f.write("[")
        for k in range(len(nn.weights2)):
            build = "["
            for l in range(len(nn.weights2[0])):
                build += str(nn.weights2[k][l])
                if l != (len(nn.weights2[0]) - 1):
                    build += ", "
            if k != (len(nn.weights2) - 1):
                build += "],\n"
            else:
                build += "]"
            f.write(build)
        f.write("]\n")
        f.write("\n")
        f.write("[")
        for k in range(len(nn.weights3)):
            build = "["
            for l in range(len(nn.weights3[0])):
                build += str(nn.weights3[k][l])
                if l != (len(nn.weights3[0]) - 1):
                    build += ", "
            if k != (len(nn.weights3) - 1):
                build += "],\n"
            else:
                build += "]"
            f.write(build)
        f.write("]\n")
    """
    while 1:
        a = int(input())
        b = a**2 + 13
        nn.train(a, b)
        X10 = 0
        for j in range(bit):
           if nn.input[0][j] == 1:
               X10 += 2**j
        outp10 = 0
        for j in range(bit):
               if nn.output[0][j] > 5.000001:
                   outp10 += 2**j
        print('prediction: ', outp10)
        print('actual answer: ', b)
"""
# tests the network with new data
testX = []
testY = []
for u in range(70):
    cont = False
    while not cont:
        a = random.randint(-1000, 1000)
        if a not in X:
            testX.append(a)
            nn.test(a)

            outp10 = 0
            for j in range(bitOut - 1):
                if nn.output[0][j] > 0.4999:
                    outp10 += 2 ** j
            if nn.output[0][bitOut - 1] > 0.4999:
                outp10 = -1 * outp10

            testY.append(outp10)
            cont = True
# draws plot
h = np.array(np.linspace(-1000, 1000, 5000))
z = np.array(19 * np.sin(h / 47) - 32 * np.cos(h / 100))

plt.plot(h, z)
plt.plot(testX, testY, "ro")
plt.xlabel("x")
plt.ylabel("Y")
plt.show()
