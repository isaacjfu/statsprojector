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
        self.CONST_FEATURES = ["heightPG","heightSF","heightC","draftPos","season","age","gp","mp","pts","reb","ast","3p","fg%","ft%","stl","blk","tov"]
        self.CONST_OUTPUT = ["gp", "mp", "pts", "reb", "ast", "3p", "fg%", "ft%", "stl", "blk", "tov"]

    def sample_data(self, name):
        f = open(self.dataFile)
        data = json.load(f)
        return self.parse_player(data,name)
    
    def parse_player(self,data,player):
        X = []
        y = []
        stats_list = data[player]['stats']
        height_string = data[player]['info']['height']
        split = height_string.split('-')
        if len(split) < 2:
            height = -1
        else:
            height = (int)(split[0])*12 + (int)(split[1])
        oharray = self.height_splitter(height)
        drafted = data[player]['info']['drafted']
        for i in range(0,len(stats_list)):
            temp_X = []
            temp_y = []
            for number in oharray:
                temp_X.append(number)
            temp_X.append((int)(drafted))
            split2 = stats_list[i]['season'].split('-')
            temp_X.append((int)(split2[0]))
            if i-1 >= 0:
                self.parse_data_helper(stats_list[i-1],temp_X,True)
            else:
                self.parse_data_helper(stats_list[i],temp_X,False)
            if i-2 >= 0:
                self.parse_data_helper(stats_list[i-2],temp_X,True)
            else:
                self.parse_data_helper(stats_list[i],temp_X,False)
            self.parse_data_helper(stats_list[i],temp_y,True,False)
            X.append(temp_X)
            y.append(temp_y)
        return (X,y)
    
    def parse_data(self):
        f = open(self.dataFile)
        data = json.load(f)
        X = []
        y = []
        for player in data:
            temp_X, temp_y = self.parse_player(data,player)
            X += temp_X
            y += temp_y
        return (X,y)

    def height_splitter(self, height):
        onehotarray = [0,0,0]
        if height == -1:
            return onehotarray
        elif height <= 76:
            onehotarray[0] = 1
        elif height > 76 and height <=81:
            onehotarray[1] = 1
        else:
            onehotarray[2] = 1
        return onehotarray
    
    def parse_data_helper(self, data,X,is_Season, is_X = True):
        if is_X:
            if is_Season:
                for i in range(5, len(self.CONST_FEATURES)):
                    X.append(data[self.CONST_FEATURES[i]])
            else:
                for i in range(5,len(self.CONST_FEATURES)):
                    X.append(0)
        else:
            for feature in self.CONST_OUTPUT:
                X.append(data[feature])

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(29,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,11)
        )
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        #print(f"logit size {logits}")
        return logits

class statsDataset(Dataset):
    def __init__(self,json_file, transform =None):
        self.transform = transform

def normalize(X,y):
    """
    X = torch.transpose(X,0,1)
    X_mean = []
    X_std = []
    for i in range(0,len(X)):
        row = X[i]
        mean = row.mean()
        std = row.std()
        X_mean.append(mean)
        X_std.append(std)
        new_row = (row-mean)/std
        X[i] = new_row
    X = torch.transpose(X,0,1)

    y = torch.transpose(y,0,1)
    y_mean = []
    y_std = []
    for i in range(0,len(y)):
        row = y[i]
        mean = row.mean()
        std = row.std()
        new_row = (row-mean)/std
        y_mean.append(mean)
        y_std.append(std)
        y[i] = new_row

    y_mean = torch.Tensor(y_mean)
    y_std = torch.Tensor(y_std)
    X_mean = torch.Tensor(X_mean)
    X_std = torch.Tensor(X_std)
    y = torch.transpose(y,0,1)
    """
    X_mean = X.mean(axis=0, keepdim=True)
    X_std = X.std(axis=0, keepdim=True)
    X = (X - X_mean) / X_std
    y_mean = y.mean(axis=0, keepdim=True)
    y_std = y.std(axis=0, keepdim=True)
    y = (y - y_mean) / y_std
    return (X,X_mean,X_std,y,y_mean,y_std)

def train():
    stats = parseData()
    X_list,y_list = stats.parse_data()
    X = torch.Tensor(X_list)
    y = torch.Tensor(y_list)
    X,X_mean,X_std,y,y_mean,y_std = normalize(X,y)
    partition = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:partition], X[partition:], y[:partition], y[partition:]

    model = NeuralNetwork()
    epochs = 100
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
                print('X:', X_train[0])
                print('y:', y_pred[0])
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | MSE Train Loss: {loss} | MSE Test Loss: {test_loss} ")
    
    MODEL_PATH = Path("models")  
    MODEL_PATH.mkdir(parents = True, exist_ok = True)
    MODEL_NAME = "statsprojector.pth"
    MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME
    torch.save(obj = model.state_dict(), f= MODEL_SAVE_PATH)

def use():
    torch.set_printoptions(sci_mode=False)
    loaded_model = NeuralNetwork()
    MODEL_PATH = Path("models")
    MODEL_NAME = "statsprojector.pth"
    loaded_model.load_state_dict(torch.load(f=MODEL_PATH/MODEL_NAME))
    stats = parseData()
    X_total, y_total = stats.parse_data()
    X_total = torch.tensor(X_total)
    y_total = torch.tensor(y_total)
    X,X_mean,X_std,y,y_mean,y_std = normalize(X_total,y_total)
    X_list, y_list = stats.sample_data('Nic Claxton')

    X = torch.Tensor(X_list)
    y = torch.Tensor(y_list)
    X = ( X - X_mean) / X_std

    loaded_model.eval()
    with torch.inference_mode():
        loaded_model_preds = loaded_model(X)
        print('X:', X)
        print('y:', loaded_model_preds)
        loaded_model_preds = (loaded_model_preds * y_std) + y_mean
    for i in range(len(y)):
        print("Season: " + str(i))
        print(loaded_model_preds[i])
        print(y[i])
        print("\n")

def testing():  
    stats = parseData()
    X_list, y_list = stats.parse_data()
    X = torch.Tensor(X_list)
    X = torch.transpose(X,0,1)
    print(X.shape)
    for i in range(0,len(X)):
        row = X[i]
        mean = row.mean()
        std = row.std()
        new_row = (row-mean)/std
        X[i] = new_row
    X = torch.transpose(X,0,1)
    print(X[:10])


if len(sys.argv) <= 1:
    print("Please put a number between 1 and 2")
else:
    if sys.argv[1] == "1":
        train()
    if sys.argv[1] == "2":
        use()
    if sys.argv[1] == '3':
        testing()