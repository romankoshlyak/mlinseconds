# The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/
#
# In this problem you need to implement model that will learn to recognize
# handwritten digits
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

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bn2d2 = nn.BatchNorm2d(5)
        self.bn2d3 = nn.BatchNorm2d(20)
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2 = nn.Conv2d(5, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=2)
        self.bn1 = nn.BatchNorm1d(160)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc1 = nn.Linear(160, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = (x-0.1215)/0.3011
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.bn2d2(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.bn2d3(x)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size()[0], -1)
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def calcModelGradNorm(self):
        model = self
        grads = [param.grad.data.view(-1) for param in model.parameters() if param.grad is not None]
        return torch.cat(grads).abs().mean()

class Solution():
    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.03:
                break
            # Put model in train mode
            model.train()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            # Wrap tensor, so we can calculate deriviative
            #printHint("Hint[3]: Experement with approximating deriviative based on subset of data", step)
            batch_size = 64
            batches = train_target.size()[0]/batch_size
            batchId = step%batches
            startInd = batchId * batch_size
            endInd = startInd + batch_size
            data = train_data[startInd:endInd,:,:,:]
            target = train_target[startInd:endInd]
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            #printHint("Hint[4]: Experement with other activation fuctions", step)
            output = model(data)
            # get the index of the max probability
            predict = output.data.max(1, keepdim=True)[1]
            # Number of correct predictions
            correct = predict.eq(target.data.view_as(predict))
            correct = correct.long()
            correct = correct.sum()
            # Total number of needed predictions
            total = target.data.view(-1).size()[0]
            # calculate loss
            loss = F.nll_loss(output, target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            self.printStats(step, loss, correct, total)
            # update model: model.parameters() -= lr * gradient
            # faster in the beginning, slower in the end
            lr = 0.002 * max(time_left, 0.3)
            # we always move same distance, this should be more stable
            lr=lr/max(model.calcModelGradNorm(), 1e-7)
            optimizer = optim.SGD(model.parameters(), lr=lr)
            optimizer.step()
            step += 1
        return step
    
    def printStats(self, step, loss, correct, total):
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
        self.test_limit = 0.95

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10
        print("Start data loading...")
        train_dataset = torchvision.datasets.MNIST(
            './../data/data_mnist', train=True, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
        test_dataset = torchvision.datasets.MNIST(
            './../data/data_mnist', train=False, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
        self.train_data = next(iter(trainLoader))
        self.test_data = next(iter(test_loader))
        print("Data loaded")

    def select_data(self, data, digits):
        data, target = data
        mask = target == -1
        for digit in digits:
            mask |= target == digit
        indices = torch.arange(0,mask.size(0))[mask].long()
        return (torch.index_select(data, dim=0, index=indices), target[mask])

    def create_case_data(self, case):
        if case == 1:
            digits = [0,1]
        elif case == 2:
            digits = [8, 9]
        else:
            digits = [i for i in range(10)]

        description = "Digits: "
        for ind, i in enumerate(digits):
            if ind > 0:
                description += ","
            description += str(i)
        train_data = self.select_data(self.train_data, digits)
        test_data = self.select_data(self.test_data, digits)
        return sm.CaseData(case, Limits(), train_data, test_data).set_description(description).set_output_size(10)

class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
