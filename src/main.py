import sys
import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from pathlib import Path
import wandb
import random
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
        # self.outputxFile = './inputFeatures.json'
        # self.outputyFile = './y_pred.json'
        self.CONST_SEASONS = 2
        self.CONST_FEATURES = ["heightPG","heightSF","heightC","draftPos","season","age","gp","mp","pts","reb","ast","3p","fg%","ft%","stl","blk","tov"]
        self.CONST_OUTPUT = ["gp", "mp", "pts", "reb", "ast", "3p", "fg%", "ft%", "stl", "blk", "tov"]

    def sample_data(self, name):
        f = open(self.dataFile)
        data = json.load(f)
        return self.parse_player(data,name)
    
    def parse_player(self,data,player):
        x = []
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
            temp_x = []
            temp_y = []
            for number in oharray:
                temp_x.append(number)
            temp_x.append((int)(drafted))
            split2 = stats_list[i]['season'].split('-')
            temp_x.append((int)(split2[0]))
            if i-1 >= 0:
                self.parse_data_helper(stats_list[i-1],temp_x,True)
            else:
                self.parse_data_helper(stats_list[i],temp_x,False)
            if i-2 >= 0:
                self.parse_data_helper(stats_list[i-2],temp_x,True)
            else:
                self.parse_data_helper(stats_list[i],temp_x,False)
            self.parse_data_helper(stats_list[i],temp_y,True,False)
            x.append(temp_x)
            y.append(temp_y)
        return (x,y)
    
    def parse_data(self):
        f = open(self.dataFile)
        data = json.load(f)
        x = []
        y = []
        for player in data:
            temp_x, temp_y = self.parse_player(data,player)
            x += temp_x
            y += temp_y
        return (x,y)

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
    
    def parse_data_helper(self, data,x,is_Season, is_x = True):
        if is_x:
            if is_Season:
                for i in range(5, len(self.CONST_FEATURES)):
                    x.append(data[self.CONST_FEATURES[i]])
            else:
                for i in range(5,len(self.CONST_FEATURES)):
                    x.append(0)
        else:
            for feature in self.CONST_OUTPUT:
                x.append(data[feature])

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(29,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,11)
        )
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class statsDataset(Dataset):
    def __init__(self,json_file, transform =None):
        self.transform = transform

def normalize(x,y):
    x_mean = x.mean(axis=0, keepdim=True)
    x_std = x.std(axis=0, keepdim=True)
    x = (x - x_mean) / x_std
    y_mean = y.mean(axis=0, keepdim=True)
    y_std = y.std(axis=0, keepdim=True)
    y = (y - y_mean) / y_std
    return (x,x_mean,x_std,y,y_mean,y_std)

def train_helper(epochs, x_train, x_test, y_train, y_test, model, lr, MODEL_NAME, x_mean, x_std):
    train_loss_values = []
    test_loss_values = []
    epoch_count = []
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)

    for epoch in range(epochs):
        model.train()
        y_pred = model(x_train)
        loss = loss_fn(y_pred,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        lossPG, lossSF, lossC, lossRookie, lossYoung, lossMid, lossOld = [0,0] , [0,0], [0,0], [0,0], [0,0], [0,0] ,[0,0]
        with torch.inference_mode():
            test_pred = model(x_test)
            test_loss = loss_fn(test_pred, y_test.type(torch.float))
            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | MSE Train Loss: {loss} | MSE Test Loss: {test_loss} ")
            
            for i in range(len(x_test)):
                loss = loss_fn(test_pred[i], y_test[i])
                if (x_test[i][0]) > 0:
                    lossPG[0] += loss
                    lossPG[1] += 1
                elif x_test[i][1] > 0:
                    lossSF[0] += loss
                    lossSF[1] += 1
                elif x_test[i][2] >0 :
                    lossC[0] += loss
                    lossC[1] += 1
                if (x_test[i][5]* x_std[0][5]) + x_mean[0][5] <= 22:
                    lossRookie[0] += loss
                    lossRookie[1] += 1
                elif (x_test[i][5]* x_std[0][5]) + x_mean[0][5] <= 29 and (x_test[i][5]* x_std[0][5]) + x_mean[0][5] > 22:
                    lossYoung[0] += loss
                    lossYoung[1] += 1
                elif (x_test[i][5]* x_std[0][5]) + x_mean[0][5] <= 35 and (x_test[i][5]* x_std[0][5]) + x_mean[0][5] > 29:
                    lossMid[0] += loss
                    lossMid[1] += 1
                else:
                    lossOld[0] += loss
                    lossOld[1] += 1

            wandb.log({"lossPG": lossPG[0]/lossPG[1], "lossSF": lossSF[0]/lossSF[1], "lossC":lossC[0]/lossC[1], "lossYoung": lossYoung[0]/lossYoung[1],
                      "lossRookie":lossRookie[0]/lossRookie[1], "lossOld" : lossOld[0]/lossOld[1], "totalLoss" : test_loss, "lossMid" : lossMid[0]/lossMid[1] })
                     
                
    
    MODEL_PATH = Path("models")  
    MODEL_PATH.mkdir(parents = True, exist_ok = True)
    MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME
    torch.save(obj = model.state_dict(), f= MODEL_SAVE_PATH)


def train():
    stats = parseData()
    x_list,y_list = stats.parse_data()
    x = torch.Tensor(x_list)
    a = x
    y = torch.Tensor(y_list)
    x,x_mean,x_std,y,y_mean,y_std = normalize(x,y)
    partition = int(len(x) * 0.8)
    x_train, x_test, y_train, y_test = x[:partition], x[partition:], y[:partition], y[partition:]
    metadata = {
        "x_mean": x_mean,
        "x_std" : x_std,
        "y_mean" : y_mean,
        "y_std" : y_std
    }
    
    model = NeuralNetwork()
    train_helper(200, x_train, x_test, y_train, y_test, model, 0.01,"statsprojector.pth", x_mean, x_std )

    PATH = Path("metadata")
    torch.save(metadata, PATH/"meanstd.pt")
    

def singleUse(firstName, lastName):
    torch.set_printoptions(sci_mode=False)
    loaded_model = NeuralNetwork()
    MODEL_PATH = Path("models")
    MODEL_NAME = "statsprojector.pth"
    loaded_model.load_state_dict(torch.load(f=MODEL_PATH/MODEL_NAME))
    stats = parseData()
    loaded_metadata = torch.load(f="metadata/meanstd.pt")
    x_mean,x_std,y_mean,y_std = loaded_metadata["x_mean"], loaded_metadata["x_std"],loaded_metadata["y_mean"],loaded_metadata["y_std"]
    x_list, y_list = stats.sample_data(firstName + ' ' + lastName)

    x = torch.Tensor(x_list)
    y = torch.Tensor(y_list)
    x = ( x - x_mean) / x_std

    loaded_model.eval()
    with torch.inference_mode():
        loaded_model_preds = loaded_model(x)
        print('x:', x)
        print('y:', loaded_model_preds)
        loaded_model_preds = (loaded_model_preds * y_std) + y_mean
    for i in range(len(y)):
        print("Season: " + str(i))
        print(loaded_model_preds[i])
        print(y[i])
        print("\n")

def testing():  
    torch.set_printoptions(sci_mode=False)
    loaded_metadata = torch.load(f="metadata/meanstd.pt")
    print(loaded_metadata["x_mean"])
    # # simulate training
    # epochs = 100
    # offset = random.random() / 5
    # for epoch in range(2, epochs):
    #     acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    #     loss = 2 ** -epoch + random.random() / epoch + offset
        
    #     # log metrics to wandb
    #     wandb.log({"acc": acc, "loss": loss})
        
    # # [optional] finish the wandb run, necessary in notebooks

if len(sys.argv) <= 1:
    print("Please put a number between 1 and 2")
else:
    if sys.argv[1] == "1":
        run = wandb.init(
            project="statsprojector",
            config={
            "learning_rate": 0.01,
            "epochs": 100,
            }
        )
        train()
    if sys.argv[1] == "2":
        singleUse(sys.argv[2], sys.argv[3])
    if sys.argv[1] == '3':
        testing()
wandb.finish()