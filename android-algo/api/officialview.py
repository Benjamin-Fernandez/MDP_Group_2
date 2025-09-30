from flask import Flask, send_file, jsonify
import os
import json

app = Flask(__name__)
IMAGE_PATH = "latest_path.png"
RAW_PATH   = "latest_response.json"

@app.route("/visualize")
def visualize():
    if not os.path.exists(IMAGE_PATH):
        return "⚠️ No visualization available yet", 400
    return send_file(IMAGE_PATH, mimetype="image/png")

@app.route("/raw")
def raw():
    if not os.path.exists(RAW_PATH):
        return jsonify({"error": "No response yet"})
    with open(RAW_PATH) as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
