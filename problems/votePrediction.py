# There are random function from 8 inputs.
# There are random input vector of size 8 * number of voters.
# We calculate function number of voters times and sum result.
# We return 1 if sum > voters/2, 0 otherwise
# We split data in 2 parts, on first part you will train and on second
# part we will test
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

class Debug(nn.Module):
    DEBUG_ON = True
    def __init__(self, model, name=None):
        super(Debug, self).__init__()
        self.model = model
        if name is None:
            self.name = "{}({:0x})".format(type(model).__name__, id(model))
        else:
            self.name = name
        self.forwardBeforeInfo = False
        self.forwardAfterInfo = True
        self.backwardBeforeInfo = False
        self.backwardAfterInfo = True
        if Debug.DEBUG_ON:
            self.register_forward_hook(self.printForwardInfo)
            self.register_backward_hook(self.printBackwardInfo)

    @classmethod
    def set_enabled(self, enabled):
        Debug.DEBUG_ON = enabled
    def setForwardBeforeInfo(self, forwardBeforeInfo):
        self.forwardBeforeInfo = forwardBeforeInfo
        return self

    def setForwardAfterInfo(self, forwardAfterInfo):
        self.forwardAfterInfo = forwardAfterInfo
        return self

    def setBackwardBeforeInfo(self, backwardBeforeInfo):
        self.backwardBeforeInfo = backwardBeforeInfo
        return self

    def setBackwardAfterInfo(self, backwardAfterInfo):
        self.backwardAfterInfo = backwardAfterInfo
        return self

    def forward(self, x):
        return self.model(x)

    def printBatchTensorInfo(self, intro, tensor):
        x = tensor
        xMean = x.view(-1).mean()
        xStd = x.std(dim=0)
        xStdMean = xStd.mean()
        xStdStd = xStd.std()
        print("{} {} {} {} {}".format(intro, xMean, xStdMean, xStdStd, self.name))

    def printForwardInfo(self, module, dataInput, dataOutput):
        if self.forwardBeforeInfo:
            self.printBatchTensorInfo("Forward In  ", dataInput[0].data)

        if self.forwardAfterInfo:
            self.printBatchTensorInfo("Forward Out ", dataOutput.data)

    def nnone(self, x):
        if x is None:
            x = torch.Tensor(1)
        return x

    def printBackwardInfo(self, module, gradInput, gradOutput):
        if self.backwardAfterInfo:
            print(len(gradInput), len(gradOutput))
            print(self.nnone(gradInput[0]).size(), self.nnone(gradInput[1]).size(), self.nnone(gradInput[2]).size())
            print(gradOutput[0].size())
            self.printBatchTensorInfo("Backward Out ", gradOutput[0].data)

        if self.backwardBeforeInfo:
            if len(gradInput) > 0 and gradInput[0] is not None:
                self.printBatchTensorInfo("Backward In  ", gradInput[0].data)

class BatchInitLinear(nn.Linear):
    def __init__(self, fromSize, toSize, solution):
        super(BatchInitLinear, self).__init__(fromSize, toSize)
        self.solution = solution
        if solution.init_type == 'uniform':
            nn.init.uniform_(self.weight, a=-1.0, b=1.0)
            nn.init.uniform_(self.bias, a=-1.0, b=1.0)
        elif solution.init_type == 'normal':
            nn.init.normal_(self.weight, 0.0, 1.0)
            nn.init.normal_(self.bias, 0.0, 1.0)
        else:
            raise "Error"
        nn.init.constant_(self.bias, 0.0)
        self.first_run = True

    def forward(self, x):
        if not self.first_run:
            return super(BatchInitLinear, self).forward(x)
        else:
            self.first_run = False
            res = super(BatchInitLinear, self).forward(x)
            resStd = res.data.std(dim=0)
            self.weight.data /= resStd.view(resStd.size(0), 1).expand_as(self.weight)
            res.data /= resStd
            if self.bias is not None:
                self.bias.data /= resStd
                resMean = res.data.mean(dim=0)
                self.bias.data -= resMean
                res.data -= resMean

            return res

class BaseModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, layers_count, solution):
        super(BaseModel, self).__init__()
        layers_size = [input_size] + [hidden_size for i in range(layers_count-1)] + [output_size]
        self.modules = []

        for ind, (from_size, to_size) in enumerate(zip(layers_size, layers_size[1:])):
            linear = Debug(BatchInitLinear(from_size, to_size, solution))
            self.add_module(str(len(self.modules)), linear)
            self.modules.append(linear)
            if ind == layers_count-1:
                self.modules.append(F.sigmoid)
            else:
                self.modules.append(F.relu)

    def forward(self, x):
        for m in self.modules:
            x = m(x)
        return x

class RandFunctionModel(BaseModel):
    def __init__(self, input_size, output_size, solution):
        super(RandFunctionModel, self).__init__(input_size, output_size, solution.rand_function_hidden_size, solution.rand_function_layers_count, solution)


class CompareModel(BaseModel):
    def __init__(self, input_size, output_size, solution):
        super(CompareModel, self).__init__(input_size, output_size, solution.compare_hidden_size, solution.compare_layers_count, solution)

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.voter_input = 8
        self.voter_count = input_size//self.voter_input
        self.solution = solution
        self.signal_count = solution.signal_count
        self.rand_function = RandFunctionModel(self.voter_input, self.signal_count, solution)
        self.compare_model = CompareModel(self.signal_count*self.voter_count, output_size, solution)

    def forward(self, x):
        x = x.view(x.size(0)*self.voter_count, -1)
        x = self.rand_function(x)
        x = x.view(x.size(0)//self.voter_count, -1)
        return self.compare_model(x)

class Solution():
    def __init__(self):
        self.learning_rate = 0.2
        self.momentum = 0.8
        self.signal_count = 3
        self.batch_size = 128
        self.init_type = 'uniform'
        self.rand_function_hidden_size = 28
        self.rand_function_layers_count = 5
        self.compare_hidden_size = 28
        self.compare_layers_count = 1
        self.clip_grad_limit = 100.0
        self.rand_seed = 1
        self.rand_seed_grid = [i for i in range(10)]

        self.grid_search = GridSearch(self).set_enabled(False)
        Debug.set_enabled(False)

    def create_model(self, input_size, output_size):
        torch.manual_seed(self.rand_seed)
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        batches = train_data.size(0)//self.batch_size
        goodCount = 0
        goodLimit = batches
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        while True:
            ind = step%batches
            start_ind = self.batch_size * ind
            end_ind = start_ind + self.batch_size
            data = train_data[start_ind:end_ind]
            target = train_target[start_ind:end_ind]
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                break
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(data).view(-1)
            target = target.view(-1)
            # if x < 0.5 predict 0 else predict 1
            predict = output.round()
            # Total number of needed predictions
            total = target.view(-1).size(0)
            # calculate loss
            bce_loss = nn.BCELoss()
            loss = bce_loss(output, target)
            diff = (output.data-target.data).abs()
            # Number of correct predictions
            correct = (diff < 0.5).long().sum().item()
            if diff.max() < 0.3:
                goodCount += 1
                if goodCount >= goodLimit:
                    break
            else:
                goodCount = 0
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            self.print_stats(step, loss, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1

        return step
    
    def print_stats(self, step, loss, correct, total):
        if step % 100 == 0:
            print("Step = {} Prediction = {}/{} Error = {}".format(step, correct, total, loss.item()))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def get_index(self, tensor_index):
        index = 0
        for i in range(tensor_index.size(0)):
            index = 2*index + tensor_index[i].item()
        return index

    def calc_value(self, input_data, function_table, input_size, input_count_size):
        count = 0
        for i in range(input_count_size):
            count += function_table[self.get_index(input_data[i*input_size: (i+1)*input_size])].item()
        if count > input_count_size/2:
            return 1
        else:
            return 0

    def create_data(self, data_size, input_size, input_count_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_table = torch.ByteTensor(function_size).random_(0, 2)
        total_input_size = input_size*input_count_size
        data = torch.ByteTensor(data_size, total_input_size).random_(0, 2)
        target = torch.ByteTensor(data_size)
        for i in range(data_size):
            target[i] = self.calc_value(data[i], function_table, input_size, input_count_size)
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        input_size = 8
        data_size = (1<<input_size)*32
        input_count_size = case

        data, target = self.create_data(2*data_size, input_size, input_count_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs per voter and {} voters".format(input_size, input_count_size))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
