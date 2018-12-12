# You need to learn a function with n inputs.
# For given number of inputs, we will generate random function.
# Your task is to learn it
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
        self.input_size = input_size
        self.output_size = output_size
        self.solution = solution
        # different seed for removing noise
        if self.solution.grid_search.enabled:
            torch.manual_seed(solution.random)
        self.hidden_size = self.solution.hidden_size
        self.linears = nn.ModuleList([nn.Linear(self.input_size if i == 0 else self.hidden_size, self.hidden_size if i != self.solution.layers_number -1 else self.output_size) for i in range(self.solution.layers_number)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.hidden_size if i != self.solution.layers_number-1 else self.output_size, track_running_stats=False) for i in range(self.solution.layers_number)])

    def forward(self, x):
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            if self.solution.do_batch_norm:
                x = self.batch_norms[i](x)
            act_function = self.solution.activation_output if i == len(self.linears)-1 else self.solution.activation_hidden
            x = self.solution.activations[act_function](x)
        return x

class Solution():
    def __init__(self):
        self = self
        self.best_step = 1000
        self.activations = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'rrelu0103': nn.RReLU(0.1, 0.3),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'leakyrelu01': nn.LeakyReLU(0.1)
        }
        self.learning_rate = 0.003
        self.momentum = 0.8
        self.layers_number = 5
        self.hidden_size = 50
        self.activation_hidden = 'relu'
        self.activation_output = 'sigmoid'
        self.do_batch_norm = True
        self.sols = {}
        self.solsSum = {}
        self.random = 0
        #self.do_batch_norm_grid = [False, True]
        self.random_grid = [_ for _ in range(10)]
        #self.layers_number_grid = [3, 4, 5, 6, 7, 8, 9, 10]
        #self.hidden_size_grid = [10, 20, 30, 40, 50]
        self.momentum_grid = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        #self.learning_rate_grid = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
        #self.activation_hidden_grid = self.activations.keys()
        #self.activation_output_grid = self.activations.keys()
        self.grid_search = GridSearch(self)
        self.grid_search.set_enabled(False)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    def get_key(self):
        return "{}_{}_{}_{}_{}_{}_{}".format(self.learning_rate, self.momentum, self.hidden_size, self.activation_hidden, self.activation_output, self.do_batch_norm, "{0:03d}".format(self.layers_number));

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        key = self.get_key()
        if key in self.sols and self.sols[key] == -1:
            return
        step = 0
        # Put model in train mode
        model.train()
        # Note: we need to move this out of circle, since we need to save state for momentum to work
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        while True:
            time_left = context.get_timer().get_time_left()
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
            if correct == total or time_left < 0.1 or (self.grid_search.enabled and step > 100):
                if not key in self.sols:
                    self.sols[key] = 0
                    self.solsSum[key] = 0
                self.sols[key] += 1
                self.solsSum[key] += step
                if self.sols[key] == len(self.random_grid):
                    print("{} {:.4f}".format(key, float(self.solsSum[key])/self.sols[key]))
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
        self.size_limit = 10000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i>>j)&1
                data[i,j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(3+case, 7)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description("{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
#Note: we need to search on case=10, since it is biggest
sm.SolutionManager(Config()).run(case_number=-1)
