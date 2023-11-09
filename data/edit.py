player_stats_file = open("rawPlayerStats.txt", "r")
new_stats_file = open("rawPlayerStats.json" , "w")
for line in player_stats_file:
    line = line.rstrip('\n')
    line = line + ',' + '\n'
    new_stats_file.write(line)
