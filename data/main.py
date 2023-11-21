from nba_api.stats.endpoints import playercareerstats, commonplayerinfo
from nba_api.stats.static import players

import json
import time
import sys
start_time = time.time()

all_players = players.get_players()
size = len(all_players)
all_stats = {}

for i in range(0, size):
    print(i)
    player = all_players[i]
    career_stats = playercareerstats.PlayerCareerStats(per_mode36 = "PerGame",
        player_id = player['id']).get_json()
    player_info = commonplayerinfo.CommonPlayerInfo(player_id = player['id']).get_json()
    career_obj = json.loads(career_stats)
    player_obj = json.loads(player_info)
    player_height = player_obj["resultSets"][0]['rowSet'][0][11]
    player_weight = player_obj["resultSets"][0]['rowSet'][0][12]
    if (player_obj["resultSets"][0]['rowSet'][0][30]) == None or (player_obj["resultSets"][0]['rowSet'][0][31]) == None or (player_obj["resultSets"][0]['rowSet'][0][30]) == 'Undrafted' or (player_obj["resultSets"][0]['rowSet'][0][31]) == 'Undrafted':
        player_drafted = -1
    else:
        player_drafted = (int)(player_obj["resultSets"][0]['rowSet'][0][30]) * (int)(player_obj["resultSets"][0]['rowSet'][0][31])
    player_name = player['full_name']
    index = 0
    reg_season_stats = []
    for i in range(len(career_obj["resultSets"])):
        if career_obj["resultSets"][i]["name"] == "SeasonTotalsRegularSeason":
            index = i
            break
    for season in career_obj["resultSets"][index]["rowSet"]:
        filtered_career_stats = {
            "team": season[4],
            "age": season[5],
            "season": season[1],
            "gp": season[6],
            "mp": season[8],
            "pts": season[26],
            "reb": season[20],
            "ast": season[21],
            "3p": season[12],
            "fg%": season[11],
            "ft%": season[17],
            "stl": season[22],
            "blk": season[23],
            "tov": season[24],
        }
        reg_season_stats.append(filtered_career_stats)
    
    player_information = {
        "height" : player_height,
        "weight" : player_weight,
        "drafted" : player_drafted
    }
    player_dict = {
        "info" : player_information,
        "stats" : reg_season_stats
    }
    all_stats[player_name] = player_dict

with open("rawPlayerStats.json", "w") as outfile:
    json.dump(all_stats,outfile,indent = 2)

print("--- %s seconds ---" % (time.time() - start_time))
