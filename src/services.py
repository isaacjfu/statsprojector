from nba_api.stats.endpoints import playercareerstats, commonplayerinfo
from nba_api.stats.static import players
import json
all_players = players.get_active_players()

def playerStatsHelper(name):
    player = players.find_players_by_full_name(name)[0]
    player_info = json.loads(commonplayerinfo.CommonPlayerInfo(player_id = player['id']).get_json())
    career_stats = playercareerstats.PlayerCareerStats(per_mode36 = "PerGame",player_id = player['id']).get_json()
    career_obj = json.loads(career_stats)
    height_string = player_info['resultSets'][0]['rowSet'][0][11]
    split = height_string.split('-')
    if len(split) < 2:
        height = -1
    else:
        height = (int)(split[0])*12 + (int)(split[1])
    oharray = height_splitter(height)
    if (player_info["resultSets"][0]['rowSet'][0][30]) == None or (player_info["resultSets"][0]['rowSet'][0][31]) == None or (player_info["resultSets"][0]['rowSet'][0][30]) == 'Undrafted' or (player_info["resultSets"][0]['rowSet'][0][31]) == 'Undrafted':
        player_drafted = -1
    else:
        player_drafted = (int)(player_info["resultSets"][0]['rowSet'][0][30]) * (int)(player_info["resultSets"][0]['rowSet'][0][31])
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
    oharray+=[player_drafted]
    url = "https://cdn.nba.com/headshots/nba/latest/1040x760/" + str(player['id']) + ".png"
    player_dict = {
        "info" : oharray,
        "image_url" : url,
        "stats" : reg_season_stats
    }
    return player_dict

def height_splitter(height):
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

def getAll():
    return all_players