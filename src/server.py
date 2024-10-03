from flask import Flask, jsonify, request
from pathlib import Path
import torch
from main import singleSeason

app = Flask(__name__)


@app.route('/predict', methods = ['GET'])
def get_season():
    data = request.get_json()
    sample = data.get('sample')
    ret = singleSeason(sample).tolist()
    return jsonify(ret)