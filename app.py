"""
NeuroCache — Flask API server
"""
from flask import Flask, jsonify, render_template, request
from dataclasses import asdict
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from simulator import run_simulation, sweep_matrix_sizes

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.get_json(force=True)
    mat_size       = int(data.get("mat_size", 64))
    cache_kb       = int(data.get("cache_kb", 32))
    cache_line     = int(data.get("cache_line", 64))
    pattern        = data.get("pattern", "row")

    result = run_simulation(mat_size, cache_kb, cache_line, pattern)
    sweep  = sweep_matrix_sizes(cache_kb, cache_line, ["row", "col", "tile", "zigzag"])

    return jsonify({
        "sim": asdict(result),
        "sweep": sweep,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
