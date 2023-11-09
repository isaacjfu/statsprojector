import json

dataFile = './rawPlayerStats.json'
outputXFile = './inputFeatures.json'
outputyFile = './y_pred.json'
CONST_SEASONS = 2
CONST_FEATURES = ["age","gp","mp","pts","reb","ast","3p","fg%","ft%","stl","blk","tov"]

def parse_data(seasons):
    f = open(dataFile)
    data = json.load(f)
    X = []
    y = []
    for i in range(seasons,len(data)):
        temp_X = []
        temp_y = []
        if data[i]["name"] == data[i-1]["name"]:
            parse_data_helper(data[i-1],temp_X,True)
        else:
            parse_data_helper(data[i-1],temp_X,False)
        if data[i]["name"] == data[i-2]["name"]:
            parse_data_helper(data[i-2],temp_X,True)
        else:
            parse_data_helper(data[i-2],temp_X,False)
        parse_data_helper(data[i],temp_y,True)
        X.append(temp_X)
        y.append(temp_y)
    return (X,y)

def parse_data_helper(data,X,is_Season):
    if is_Season:
        for feature in CONST_FEATURES:
            X.append(data[feature])
    else:
        for i in range(0,len(CONST_FEATURES)):
            X.append(-1)

def writeToJson(X,y):
    inputFeatures = open(outputXFile, "w")
    inputFeatures.write("[")
    inputFeatures.write("]")

    y_pred = open(outputyFile,"w")
    y_pred.write("[")
    y_pred.write("]")

X,y = parse_data(CONST_SEASONS)
writeToJson(X,y)
