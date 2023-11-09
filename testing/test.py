from nba_api.stats.endpoints import playercareerstats

# Nikola Jokić
career = playercareerstats.PlayerCareerStats(player_id='203999') 


# json
career.get_json()

# dictionary
career.get_dict()

print(career.get_json())