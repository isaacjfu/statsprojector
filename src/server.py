from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import torch
from main import singleSeason
from services import playerStatsHelper, getAll

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods = ['POST'])
def get_season():
    data = request.get_json()
    sample = data.get('sample')
    ret = singleSeason(sample).tolist()
    return jsonify(ret)

@app.route('/', methods = ['GET'])
def get_all_active():
    return jsonify(getAll())

@app.route('/getPlayer', methods = ['POST'])
def get_player_stats():
    data = request.get_json()
    name = data.get('name')
    return jsonify(playerStatsHelper(name))

@app.errorhandler(404)
def page_not_found(e):
    return jsonify(error="Endpoint not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify(error="Internal server error"), 500