import sys
import json

if len(sys.argv) <= 1:
    print("Please put a number between 1 and 2")
else:
    #add commas for conversion to json file
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
        for player in data:
            stats_list = data[player]['stats']
            size = len(stats_list)
            r_list = []
            i = 0
            while i < size:
                pos = size-1-i
                getRidOfNulls(stats_list[i])
                if(stats_list[pos]["team"] == "TOT"):
                    age = stats_list[pos]["age"]
                    j = pos -1
                    while (stats_list[j]["age"] == age):
                        r_list.append(stats_list[j])
                        j -= 1
                        if j < 0:
                            break
                i += 1
            for item in r_list:
                stats_list.remove(item)

    
    def getRidOfNulls(data):
        for attribute,value in data.items():
            if value == None:
                data[attribute] = 0.0

    # combine tot seasons for players and get rid of nulls
    if sys.argv[1] == "2":
        new_stats_file = open("playerStats.json","w")
        f = open('rawPlayerStats.json')
        data = json.load(f)
        combineTotSeasons(data)
        json.dump(data,new_stats_file,indent = 2)
        f.close()
        new_stats_file.close()
 