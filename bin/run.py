#!/usr/bin/env python
from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.base.base_summarizer import get_summary
app = Flask(__name__)
CORS(app)



@app.route('/resume/<int:summary_percent>/<string:path>', methods=['GET'])
def resume(summary_percent,path):
    print(path)
    #summary_percent = 25
    summary_sens = get_summary(path, summary_percent, model_path=r'D:\An2Master\topici\wiki-news-300d-1M.vec\wiki-news-300d-1M.vec') #model_path for word2Vec model
    result = " ".join(summary_sens)
    return jsonify({'resume': result}), 200

if __name__ == '__main__':
    app.run(debug=True)
    # model_path = sys.argv[1]
    # print("Going to load word2Vec Model, it may take a minute or more")
    #
    # print("model loaded, now going to quickly generate summary")
    #
    # summary_percent = 5
    # input_file = sys.argv[2]
    # summary_sens = get_summary(input_file, summary_percent, model_path=model_path)
    # for s in summary_sens:
    #     print(s)
