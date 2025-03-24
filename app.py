# app.py
from flask import Flask, render_template, request, jsonify
import polars as pl
import data_prep as data_prep
from model.binning_input_games import bin_board_games
from model.cluster_games import bgClusters
from model.recommend_games import RecommendationEngine
import json
import os
import datetime
import pandas as pd

app = Flask(__name__)

# Load and prepare data once at startup
df = pl.read_parquet("bgg/data/raw_bgg_data_20250216.parquet")
df = data_prep.run_data_preparation(df)

# Set up the model
constraint_columns = ['GAME_CAT_GROUP_card_game']
model = bgClusters()
clusters = model.fit(df, constraint_columns=constraint_columns, name_column="game_name")

# Create recommendation engine
rec = RecommendationEngine()

# Ensure feedback directory exists
FEEDBACK_DIR = "feedback"
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "user_feedback.csv")

if not os.path.exists(FEEDBACK_DIR):
    os.makedirs(FEEDBACK_DIR)

# Create feedback file with headers if it doesn't exist
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=[
        'timestamp', 'input_game', 'recommended_game', 'feedback'
    ]).to_csv(FEEDBACK_FILE, index=False)

@app.route('/')
def index():
    # Get a list of all game names for autocomplete
    game_names = df["game_name"].to_list()
    return render_template('index.html', game_names=game_names)

@app.route('/recommend', methods=['POST'])
def recommend():
    game_name = request.form['game_name']
    
    try:
        recommendations = rec.recommend_games(clusters.clusters, game_name, df)
        
        if "error" in recommendations:
            return jsonify({"error": recommendations["error"]})
        
        result = {
            "input_game": recommendations["input_game"],
            "recommendations": []
        }
        
        for game in recommendations["recommended_games"]:
            game_scores = recommendations["recommendation_scores"][game]
            explanation = rec.explain_recommendation(game_scores)
            result["recommendations"].append({
                "name": game,
                "explanation": explanation
            })
            
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/save_feedback', methods=['POST'])
def save_feedback():
    try:
        feedback_data = request.json
        
        # Create a DataFrame with the feedback
        feedback_df = pd.DataFrame([{
            'timestamp': feedback_data['timestamp'],
            'input_game': feedback_data['input_game'],
            'recommended_game': feedback_data['recommended_game'],
            'feedback': feedback_data['feedback']
        }])
        
        # Append to the CSV file
        feedback_df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
        
        return jsonify({"status": "success", "message": "Feedback saved successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)