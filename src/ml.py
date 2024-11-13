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
    def __init__(self,seasons: int= 0):
        self.dataFile = '../data/playerStats.json'
        # self.outputxFile = './inputFeatures.json'
        # self.outputyFile = './y_pred.json'
        self.CONST_SEASONS = seasons
        self.CONST_STATIC_FEATURES = ["heightPG","heightSF","heightC","draftPos","season","age"]
        self.CONST_FEATURES = ["gp","mp","pts","reb","ast","3p","fg%","ft%","stl","blk","tov"]
        self.CONST_OUTPUT = ["gp", "mp", "pts", "reb", "ast", "3p", "fg%", "ft%", "stl", "blk", "tov"]

    def returnOutput(self):
        return self.CONST_OUTPUT
    
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
            temp_x.append((int)(stats_list[i]['age']))
            for j in range(1,self.CONST_SEASONS+1):
                if(i-j >= 0):
                    self.parse_data_helper(stats_list[i-j],temp_x, True, True)
                else:
                    self.parse_data_helper(stats_list[i],temp_x, False, True)
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
    
    def parse_data_helper(self, data, x, is_Season, is_x = True):
        if is_x:
            if is_Season:
                for i in range(0, len(self.CONST_FEATURES)):
                    x.append(data[self.CONST_FEATURES[i]])
            else:
                for i in range(0,len(self.CONST_FEATURES)):
                    x.append(0)
        else:
            for feature in self.CONST_OUTPUT:
                x.append(data[feature])

class NeuralNetworkSmall(nn.Module):
    def __init__(self, seasons):
        super().__init__()
        self.flatten = nn.Flatten()
        inputs = 6 + (11*seasons)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputs,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,11)
        )
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
class NeuralNetworkBig(nn.Module):
    def __init__(self, seasons):
        super().__init__()
        self.flatten = nn.Flatten()
        inputs = 6 + (11*seasons)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputs,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
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
    FILTER_FUNCS = {
        'lossPG': lambda x,y,z: x[:,0:1]  > 0,
        'lossSF' : lambda x,y,z: x[:,1:2] > 0,
        'lossC': lambda x,y,z: x[:,2:3]  > 0,
        'lossRookie' : lambda x,y,z : (x[:,5:6] * y[:,5:6]) + z[:,5:6] <= 22,
        'lossYoung' : lambda x,y,z : torch.logical_and((x[:,5:6] * y[:,5:6]) + z[:,5:6] <= 29,(x[:,5:6] * y[:,5:6]) + z[:,5:6] > 22),
        'lossMid' : lambda x,y,z : torch.logical_and((x[:,5:6] * y[:,5:6]) + z[:,5:6] <= 35,(x[:,5:6] * y[:,5:6]) + z[:,5:6] > 29),
        'lossOld' : lambda x,y,z : (x[:,5:6] * y[:,5:6]) + z[:,5:6] > 35,
    }
    train_loss_values = []
    test_loss_values = []
    epoch_count = []
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
    stats = parseData()
    catNames = stats.returnOutput()
    for epoch in range(epochs):
        model.train()
        y_pred = model(x_train)
        loss = loss_fn(y_pred,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            test_pred = model(x_test)
            test_loss = loss_fn(test_pred, y_test.type(torch.float))
            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | MSE Train Loss: {loss} | MSE Test Loss: {test_loss} ")
            
            for func in FILTER_FUNCS:
                indicies = (torch.where(FILTER_FUNCS[func](x_test,x_std,x_mean), 1 , 0) != 0).any(dim=1).nonzero(as_tuple=True)[0]
                x_filter = x_test[indicies]
                y_filter = y_test[indicies]
                filter_loss = loss_fn(model(x_filter),y_filter)
                wandb.log({func : filter_loss})
            for i in range(len(catNames)):
                catLoss = loss_fn(test_pred[i:i+1], y_test[i:i+1])
                wandb.log( {catNames[i]: catLoss/len(x_test)})
            wandb.log({"totalLoss": test_loss})
        
    MODEL_PATH = Path("models")  
    MODEL_PATH.mkdir(parents = True, exist_ok = True)
    MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME
    torch.save(obj = model.state_dict(), f= MODEL_SAVE_PATH)


def train(seasons,model,file,epochs, lr):
    stats = parseData(seasons)
    x_list,y_list = stats.parse_data()
    x = torch.Tensor(x_list)
    y = torch.Tensor(y_list)
    x,x_mean,x_std,y,y_mean,y_std = normalize(x,y)
    partition = int(len(x) * 0.8)
    indicies = torch.randperm(len(x_list))
    train_split, test_split = indicies[partition:], indicies[:partition]
    x_train, y_train = x[train_split], y[train_split]
    x_test, y_test = x[test_split], y[test_split]

    train_helper(epochs, x_train, x_test, y_train, y_test, model, lr ,file, x_mean, x_std)

    metadata = {
        "x_mean": x_mean,
        "x_std" : x_std,
        "y_mean" : y_mean,
        "y_std" : y_std
    }
    PATH = Path("metadata")
    torch.save(metadata, PATH/"meanstd.pt")

# 2 is generally best amount of season based off of hyperparameter tuning
def train_main(seasons):
    for i in range(0,5):
        model_small = NeuralNetworkSmall(seasons) 
        train(seasons,model_small,'statsprojector' + str(i) + '.pth', 300, 0.005)


#hyperparameter tuning    
# def train_main(seasons):
#     for season in range(1,5):
#         model_small = NeuralNetworkSmall(season) 
#         print(str(season) + " SMALL")
#         train(season,model_small,'statsprojector' + str(season) + '.pth',200)
#         model_big = NeuralNetworkBig(season)
#         print(str(season) + " BIG")
#         train(season,model_big,'statsprojectorbig' + str(season) + '.pth',200)

def singleUse(firstName, lastName, seasons):
    torch.set_printoptions(sci_mode=False)
    stats = parseData(seasons)
    loaded_metadata = torch.load(f="metadata/meanstd.pt")
    x_mean,x_std,y_mean,y_std = loaded_metadata["x_mean"], loaded_metadata["x_std"],loaded_metadata["y_mean"],loaded_metadata["y_std"]
    x_list, y_list = stats.sample_data(firstName + ' ' + lastName)
    x = torch.Tensor(x_list)
    y = torch.Tensor(y_list)
    x = ( x - x_mean) / x_std
    y_pred_aggregrate = torch.zeros_like(y)
    batches = 5
    loss_fn = nn.MSELoss()
    loss = 0
    for i in range (0,batches):
        loaded_model = NeuralNetworkSmall(seasons)
        MODEL_PATH = Path("models")
        MODEL_NAME = "statsprojector" + (str)(i) + ".pth"
        loaded_model.load_state_dict(torch.load(f=MODEL_PATH/MODEL_NAME))
        loaded_model.eval()
        with torch.inference_mode():
            loaded_model_preds = loaded_model(x)
            loaded_model_preds = (loaded_model_preds * y_std) + y_mean
            y_pred_aggregrate = y_pred_aggregrate.add_(loaded_model_preds)
    y_pred = y_pred_aggregrate/(batches)
    for i in range(len(y)):
        print("Season:" + str(i))
        print(y_pred[i])
        print(y[i])
    loss = loss_fn(y_pred,y)
    print(loss)

def singleSeason(stats):
    stats = torch.Tensor(stats)
    loaded_metadata = torch.load(f="metadata/meanstd.pt",weights_only=True)
    x_mean,x_std,y_mean,y_std = loaded_metadata["x_mean"], loaded_metadata["x_std"],loaded_metadata["y_mean"],loaded_metadata["y_std"]
    x = ( stats - x_mean) / x_std
    y_pred_aggregrate = torch.zeros_like(y_mean)
    batches = 5
    for i in range (0,batches):
        loaded_model = NeuralNetworkSmall(2)
        MODEL_PATH = Path("models")
        MODEL_NAME = "statsprojector" + (str)(i) + ".pth"
        loaded_model.load_state_dict(torch.load(f=MODEL_PATH/MODEL_NAME, weights_only=True))
        loaded_model.eval()
        with torch.inference_mode():
            loaded_model_preds = loaded_model(x)
            loaded_model_preds = (loaded_model_preds * y_std) + y_mean
            y_pred_aggregrate = y_pred_aggregrate.add_(loaded_model_preds)
    y_pred = y_pred_aggregrate/(batches)
    return(y_pred)

def testing():
    torch.set_printoptions(sci_mode=False)
    stats = parseData(2)
    x_list, y_list = stats.sample_data('Stephen Curry')
    test = x_list[-1]
    print(test)
    #print(singleSeason(test))

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
        train_main((int)(sys.argv[2]))
    if sys.argv[1] == "2":
        singleUse(sys.argv[2], sys.argv[3], (int)(sys.argv[4]))
    if sys.argv[1] == '3':
        testing()
wandb.finish()