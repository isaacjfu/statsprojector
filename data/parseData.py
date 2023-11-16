import json

# dataFile = './rawPlayerStats.json'
# outputXFile = './inputFeatures.json'
# outputyFile = './y_pred.json'
# CONST_SEASONS = 2
# CONST_FEATURES = ["age","gp","mp","pts","reb","ast","3p","fg%","ft%","stl","blk","tov"]

class parseData:
    def __init__(self):
        self.dataFile = './rawPlayerStats.json'
        self.outputXFile = './inputFeatures.json'
        self.outputyFile = './y_pred.json'
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
                
    def writeToJson(self, X,y):
        inputFeatures = open(self.outputXFile, "w")
        inputFeatures.write("[")
        inputFeatures.write("]")

        y_pred = open(self.outputyFile,"w")
        y_pred.write("[")
        y_pred.write("]")
