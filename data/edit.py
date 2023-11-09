player_stats_file = open("playerStatss.txt", "r")
new_stats_file = open("playerStats.json" , "w")
for line in player_stats_file:
    line = line.rstrip('\n')
    line = line + ',' + '\n'
    new_stats_file.write(line)
