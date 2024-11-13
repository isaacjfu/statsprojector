from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import torch
from ml import singleSeason
from services import playerStatsHelper, getAll

application = app = Flask(__name__)
CORS(application)

# web: gunicorn --bind 0.0.0.0:8080 server:applicationlication

@application.route('/predict', methods = ['POST'])
def get_season():
    data = request.get_json()
    sample = data.get('sample')
    ret = singleSeason(sample).tolist()[0]
    season_value = str((int)(sample[4]) + 1)[2:]
    season_value = str(sample[4]) + '-' + season_value
    filtered_proj_stats = {
        "team": "---",
        "age": sample[5],
        "season": season_value,
        "gp": round(ret[0],0), 
        "mp": round(ret[1],1),
        "pts": round(ret[2],1),
        "reb": round(ret[3],1),
        "ast": round(ret[4],1),
        "3p": round(ret[5],1),
        "fg%": round(ret[6],3),
        "ft%": round(ret[7],3),
        "stl": round(ret[8],1),
        "blk": round(ret[9],1),
        "tov": round(ret[10],1),
    }
    return jsonify(filtered_proj_stats)

@application.route('/', methods = ['GET'])
def get_all_active():
    return jsonify(getAll())

@application.route('/getPlayer', methods = ['POST'])
def get_player_stats():
    data = request.get_json()
    name = data.get('name')
    return jsonify(playerStatsHelper(name))

@application.errorhandler(404)
def page_not_found(e):
    return jsonify(error="Endpoint not found"), 404

@application.errorhandler(500)
def internal_server_error(e):
    return jsonify(error="Internal server error"), 500

if __name__ == "__main__":
    application.run()