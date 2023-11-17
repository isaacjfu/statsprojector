import sys
import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt

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
    
    def sample_data(self, name):
        f = open(self.dataFile)
        data = json.load(f)
        X = []
        y = []
        for i in range(self.CONST_SEASONS, len(data)):
            if data[i]["name"] == name:
                j = i
                while data[j]["name"] == name:
                    temp_X = []
                    temp_y = []
                    self.parse_data_helper(data[j],temp_y, True)
                    if name == data[j-1]["name"]:
                        self.parse_data_helper(data[j-1],temp_X,True)
                    else:
                        self.parse_data_helper(data[j-1],temp_X,False)
                    if name == data[j-2]["name"]:
                        self.parse_data_helper(data[j-2],temp_X,True)
                    else:
                        self.parse_data_helper(data[j-2],temp_X,False)
                    X.append(temp_X)
                    y.append(temp_y)
                    j +=1
                break
        return (X,y)

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
        #print(f"logit size {logits}")
        return logits

class statsDataset(Dataset):
    def __init__(self,json_file, transform =None):
        self.transform = transform

def runNN():
    stats = parseData()
    X_list,y_list = stats.parse_data()
    X = torch.Tensor(X_list)
    y = torch.Tensor(y_list)
    partition = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[partition:], X[:partition], y[partition:], y[:partition]
    # generator1 = torch.Generator().manual_seed(42)
    # X_train, X_test = torch.utils.data.random_split(X,[0.8,0.2],generator = generator1)    
    # y_train, y_test = torch.utils.data.random_split(y,[0.8,0.2],generator = generator1)
    print(X_train[5:])
    print(y_train[5:])

    model = NeuralNetwork()
    epochs = 200
    train_loss_values = []
    test_loss_values = []
    epoch_count = []
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_pred,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test.type(torch.float))

            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | MSE Train Loss: {loss} | MSE Test Loss: {test_loss} ")
    
    MODEL_PATH = Path("models")  
    MODEL_PATH.mkdir(parents = True, exist_ok = True)
    MODEL_NAME = "statsprojector.pth"
    MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME
    torch.save(obj = model.state_dict(), f= MODEL_SAVE_PATH)

def useNN():
    loaded_model = NeuralNetwork()
    MODEL_PATH = Path("models")
    MODEL_NAME = "statsprojector.pth"
    loaded_model.load_state_dict(torch.load(f=MODEL_PATH/MODEL_NAME))
    stats = parseData()
    X_list, y_list = stats.sample_data('Stephen Curry')
    X = torch.Tensor(X_list)
    y = torch.Tensor(y_list)
    print(X)
    print(y)
    loaded_model.eval()
    with torch.inference_mode():
        loaded_model_preds = loaded_model(X)
    for i in range(len(y)):
        print("Season: " + str(i))
        print(loaded_model_preds[i])
        print(y[i])
        print("\n")

def testing():  
    loaded_model = NeuralNetwork()
    MODEL_PATH = Path("models")
    MODEL_NAME = "statsprojector.pth"
    loaded_model.load_state_dict(torch.load(f=MODEL_PATH/MODEL_NAME))
    X_list = [[34.0,64.0,34.5,25.5,5.2,6.3,4.5,0.437,0.923,1.3,0.4,3.2,35.0,56.0,34.7,29.4,6.1,6.3,4.9,0.493,0.915,0.9,0.4,3.2]]
    X = torch.Tensor(X_list)
    loaded_model.eval()
    with torch.inference_mode():
        preds = loaded_model(X)
    print(preds)

if len(sys.argv) <= 1:
    print("Please put a number between 1 and 2")
else:
    if sys.argv[1] == "1":
        runNN()
    if sys.argv[1] == "2":
        useNN()
    if sys.argv[1] == '3':
        testing()