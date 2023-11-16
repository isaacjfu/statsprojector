import sys
import json
#add commas for conversion to json file
if len(sys.argv) <= 1:
    print("Please put a number between 1 and 2")
else:
    if sys.argv[1] == "1":
        player_stats_file = open("rawPlayerStats.txt", "r")
        new_stats_file = open("rawPlayerStats.json" , "a+")
        new_stats_file.write('[')
        count = 0
        count_two = 0
        for line in player_stats_file:
            count +=1
        player_stats_file.seek(0,0)
        for line in player_stats_file:
            count_two +=1
            line = line.rstrip('\n')
            if count_two == count:
                line = line + ']'
            else:
                line = line + ',' + '\n'
            new_stats_file.write(line)
        player_stats_file.close()
        new_stats_file.close()

    def combineTotSeasons(data):
        temp_string_list = []
        size = len(data)
        i = 0
        while i < size:
            pos = size - i -1
            getRidOfNulls(data[pos])
            temp_string_list.append(json.dumps(data[pos]))   
            if(data[pos]["team"] == "TOT"):
                name = data[pos]["name"]
                age = data[pos]["age"]
                j = pos
                while (data[j]["age"] == age) and (data[j]["name"] == name):
                    j -= 1
                    i += 1
                i -=1
            i +=1  
        return temp_string_list
    
    def getRidOfNulls(data):
        for attribute,value in data.items():
            if value == None:
                data[attribute] = 0.0

    def writeFile(temp_string_list, new_stats_file):
        for i in range(0,len(temp_string_list)):
            if i == 0:
                new_stats_file.write('[')
            pos = len(temp_string_list) - i - 1
            new_stats_file.write(temp_string_list[pos])
            if i != len(temp_string_list)-1:
                new_stats_file.write(',' + '\n')
            if i == len(temp_string_list)-1:
                new_stats_file.write(']')

    # combine tot seasons for players and get rid of nulls
    if sys.argv[1] == "2":
        new_stats_file = open("playerStats.json","w")
        f = open('rawPlayerStats.json')
        data = json.load(f)
        temp_string_list = combineTotSeasons(data)
        writeFile(temp_string_list,new_stats_file)
        f.close()
        new_stats_file.close()
 