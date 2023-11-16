import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class parseData:
    def __init__(self):
        self.dataFile = '../data/playerStats.json'
        # self.outputXFile = './inputFeatures.json'
        # self.outputyFile = './y_pred.json'
        self.CONST_SEASONS = 2
        self.CONST_FEATURES = ["age","gp","mp","pts","reb","ast","3p","fg%","ft%","stl","blk","tov"]


    def parse_data(self):
        f = open(self.dataFile)
        data = json.load(f)
        X = []
        y = []
        for i in range(self.CONST_SEASONS,len(data)):
            temp_X = []
            temp_y = []
            if data[i]["name"] == data[i-1]["name"]:
                self.parse_data_helper(data[i-1],temp_X,True)
            else:
                self.parse_data_helper(data[i-1],temp_X,False)
            if data[i]["name"] == data[i-2]["name"]:
                self.parse_data_helper(data[i-2],temp_X,True)
            else:
                self.parse_data_helper(data[i-2],temp_X,False)
            self.parse_data_helper(data[i],temp_y,True)
            X.append(temp_X)
            y.append(temp_y)
        return (X,y)

    def parse_data_helper(self, data,X,is_Season):
        if is_Season:
            for feature in self.CONST_FEATURES:
                X.append(data[feature])
        else:
            for i in range(0,len(self.CONST_FEATURES)):
                X.append(-1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(24,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,12)
        )
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        print(f"logit size {logits}")
        return logits

class statsDataset(Dataset):
    def __init__(self,json_file, transform =None):
        self.transform = transform

stats = parseData()
X_list,y_list = stats.parse_data()
print(X_list[:50])
print(type(X_list))
X = torch.Tensor(X_list)
y = torch.Tensor(y_list)
print(X.shape)
print(y.shape)
# print(X[:2])
# print(y[:2])
# model = NeuralNetwork().to(device)

#logits = model(X)
# print(logits)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class : {y_pred}")
