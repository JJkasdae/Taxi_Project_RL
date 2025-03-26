import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent:
    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.copy_weights = []
        self.copy_biases = []
        self.parameters = []
        self.optimizer = None
        
    def setUp(self):
        """
        set up the NN
        """
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(torch.randn(self.layer_sizes[i], self.layer_sizes[i+1], requires_grad=True, device=self.device))
            self.biases.append(torch.randn(self.layer_sizes[i+1], requires_grad=True, device=self.device))
            self.copy_weights.append(torch.randn(self.layer_sizes[i], self.layer_sizes[i+1], device=self.device))
            self.copy_biases.append(torch.randn(self.layer_sizes[i+1], device=self.device))
        # print(type(self.copy_weights[0]))
        # print(type(self.copy_biases[0]))
        # print("Data type: ", type(self.weights[0]), "Size is: ", self.weights[0].size())
        # print("Data type: ", type(self.biases[0]), "Size is: ", self.biases[0].size())
        for i in range(len(self.weights)):
            self.parameters.append(self.weights[i])
            self.parameters.append(self.biases[i])
        self.optimizer = torch.optim.Adam(self.parameters, lr = 0.0001)

    def copy(self):
        """
        copy the NN
        """
        for i in range(len(self.weights)):
            self.copy_weights[i] = self.weights[i].clone()
            self.copy_biases[i] = self.biases[i].clone()



    def forward(self,x, td_target = False):
        """
        predict the output in the NN
        """
        x = x.to(self.device)
        if (not td_target):
            for i in range(len(self.layer_sizes) - 1): # layer size is 3, loop in [0, 1]
                x = x@self.weights[i] + self.biases[i]
                if i < len(self.layer_sizes) - 2: # 
                    x = F.relu(x)
        else:
            for i in range(len(self.layer_sizes) - 1):
                x = x@self.copy_weights[i] + self.copy_biases[i]
                if i < len(self.layer_sizes) - 2:
                    x = F.relu(x)
        return x

# nn = NeuralNetwork([3, 10, 2])
# nn.setUp()

