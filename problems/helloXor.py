# HelloXor is a HelloWorld of Machine Learning.
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('./../utils')
import solutionmanager as sm
from gridsearch import GridSearch

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        # different seeds for random initialization for grid search
        if solution.grid_search.enabled:
            torch.manual_seed(solution.random)
        self.solution = solution
        self.input_size = input_size
        self.hidden_size = self.solution.hidden_size
        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.solution.activations[self.solution.activation_hidden](x)
        x = self.linear2(x)
        x = self.solution.activations[self.solution.activation_output](x)
        return x

class Solution():
    def htanh02(self, x):
        return nn.Hardtanh(-0.2, 0.2)(x)
    def custom(self, x):
        return self.htanh02(0.72*x)+self.htanh02(0.27*x)+self.htanh02(0.2*x)+self.htanh02(0.2*x)+self.htanh02(0.1*x)+0.2*x
    def __init__(self):
        self.best_step = 1000
        self.activations = {
                'sigmoid': nn.Sigmoid(),
                'custom': self.custom,
                'relu': nn.ReLU(),
                'relu6': nn.ReLU6(),
                'rrelu0103': nn.RReLU(0.1, 0.3),
                'rrelu0205': nn.RReLU(0.2, 0.5),
                'htang1': nn.Hardtanh(-1, 1),
                'htang2': nn.Hardtanh(-2, 2),
                'htang3': nn.Hardtanh(-3, 3),
                'tanh': nn.Tanh(),
                'elu': nn.ELU(),
                'selu': nn.SELU(),
                'hardshrink': nn.Hardshrink(),
                'leakyrelu01': nn.LeakyReLU(0.1),
                'leakyrelu001': nn.LeakyReLU(0.01),
                'logsigmoid': nn.LogSigmoid(),
                'prelu': nn.PReLU(),
            }
        self.learning_rate = 1.0
        self.hidden_size = 11
        self.activation_hidden = 'relu'
        self.activation_output = 'sigmoid'
        self.sols = {}
        self.solsSum = {}
        self.random = 0
        self.random_grid = [_ for _ in range(10)]
        self.hidden_size_grid = [3, 5, 7, 11]
        self.learning_rate_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        #self.learning_rate_grid = [1.0 + i/100.0 for i in range(10)]
        self.activation_hidden_grid = self.activations.keys()
        #self.activation_output_grid = self.activations.keys()
        self.grid_search = GridSearch(self)
        self.grid_search.set_enabled(True)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.train()
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            key = "{}_{}_{}_{}".format(self.learning_rate, self.hidden_size, self.activation_hidden, self.activation_output)
            # Speed up search
            if time_left < 0.1 or (self.grid_search.enabled and step > 40):
                if not key in self.sols:
                    self.sols[key] = 0
                    self.solsSum[key] = 0
                self.sols[key] += 1
                self.solsSum[key] += step
                self.sols[key] = -1
                break
            if key in self.sols and self.sols[key] == -1:
                break
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
            data = train_data
            target = train_target
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(data)
            # if x < 0.5 predict 0 else predict 1
            predict = output.round()
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = target.view(-1).size(0)
            if correct == total:
                if not key in self.sols:
                    self.sols[key] = 0
                    self.solsSum[key] = 0
                self.sols[key] += 1
                self.solsSum[key] += step
                #if self.sols[key] > 1:
                #    print("Key = {} Avg = {:.2f} Ins = {}".format(key, float(self.solsSum[key])/self.sols[key], self.sols[key]))
                if self.sols[key] == len(self.random_grid):
                    #self.best_step = step
                    print("Learning rate = {} Hidden size = {} Activation hidden = {} Activation output = {} Steps = {}".format(self.learning_rate, self.hidden_size, self.activation_hidden, self.activation_output, step))
                    print("{:.4f}".format(float(self.solsSum[key])/self.sols[key]))
                break
            # calculate loss
            loss = ((output-target)**2).sum()
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            #self.print_stats(step, loss, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        return step

    def print_stats(self, step, loss, correct, total):
        if step % 1000 == 0:
            print("Step = {} Prediction = {}/{} Error = {}".format(step, correct, total, loss.item()))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 100
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self):
        data = torch.FloatTensor([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
            ])
        target = torch.FloatTensor([
            [0.0],
            [1.0],
            [1.0],
            [0.0]
            ])
        return (data, target)

    def create_case_data(self, case):
        data, target = self.create_data()
        return sm.CaseData(case, Limits(), (data, target), (data, target))

class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
