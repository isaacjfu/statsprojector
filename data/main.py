from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
import json
import time

start_time = time.time()

all_players = players.get_players()
all_stats = []

player_stats_file = open("rawPlayerStats.txt", "a")
for i in range(0, len(all_players)):
    player = all_players[i]

    career_stats = playercareerstats.PlayerCareerStats(per_mode36 = "PerGame",
        player_id = player['id']).get_json()
    career_obj = json.loads(career_stats)
    player_name = player['full_name']
    index = 0

    print(player_name)
    print(i)

    for i in range(len(career_obj["resultSets"])):
        if career_obj["resultSets"][i]["name"] == "SeasonTotalsRegularSeason":
            index = i
            break
    for season in career_obj["resultSets"][index]["rowSet"]:
        filtered_career_stats = {
            "name": player_name,
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
        player_stats_file.write(json.dumps(filtered_career_stats))
        player_stats_file.write("\n")
        # all_stats.append(filtered_career_stats)
player_stats_file.close()
# with open("playerStats.json", "w") as outfile:
#     json.dump(all_stats, outfile)

print("--- %s seconds ---" % (time.time() - start_time))


# for player in all_players:
#     career_stats = playercareerstats.PlayerCareerStats(per_mode36 = "PerGame",
#         player_id = player['id']).get_json()
#     career_obj = json.loads(career_stats)
#     player_name = player['full_name']
#     index = 0
#
#     print(player_name)
#     print(player['id'])
#
#     for i in range(len(career_obj["resultSets"])):
#         if career_obj["resultSets"][i]["name"] == "SeasonTotalsRegularSeason":
#             index = i
#             break
#     for season in career_obj["resultSets"][index]["rowSet"]:
#         filtered_career_stats = {
#             "name": player_name,
#             "team": season[4],
#             "age": season[5],
#             "season": season[1],
#             "gp": season[6],
#             "mp": season[8],
#             "pts": season[26],
#             "reb": season[20],
#             "ast": season[21],
#             "3p": season[12],
#             "fg%": season[11],
#             "ft%": season[17],
#             "stl": season[22],
#             "blk": season[23],
#             "tov": season[24],
#         }
#         all_stats.append(filtered_career_stats)
#
# with open("playerStats.json", "w") as outfile:
#     json.dump(all_stats, outfile)
#
# print("--- %s seconds ---" % (time.time() - start_time))
