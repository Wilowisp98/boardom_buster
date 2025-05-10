# -*- coding: utf-8 -*-
import os
import asyncio
from datetime import datetime
from typing import Dict
import argparse

import polars as pl
from flask import Flask, render_template, request, jsonify

from src.model.cluster_games import bgClusters
from src.model.recommend_games import RecommendationEngine
from src.bgg.bgg import main_bgg
from src.data_processing.data_prep import run_data_preparation
from configs import *

app = Flask(__name__, template_folder='src/templates', static_folder='src/static')

def is_data_folder_empty(step: str = None) -> bool:
    if step == 'bgg':
        data_path = BGG_PATH
    if step == 'prep':
        data_path = PREP_PATH
    if step == 'model':
        data_path = MODEL_PATH
    if not os.path.exists(data_path):
        return True
    return len(os.listdir(data_path)) == 0

def get_latest_file(directory: str) -> str:
    try:
        files = os.listdir(directory)
        dated_files = []
        for file in files:
            try:
                date_str = file.split('_')[-1].split('.')[0]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                dated_files.append((file_date, file))
            except (ValueError, IndexError):
                continue
        
        if not dated_files:
            raise ValueError(f"No valid dated files found in {directory}")
            
        latest_file = sorted(dated_files, reverse=True)[0][1]

        return os.path.join(directory, latest_file)
    except Exception as e:
        raise ValueError(f"Error getting latest file: {e}")

def is_file_older_than_days(file_path: str, days: int = 7) -> bool:
    try:
        file_date_str = os.path.basename(file_path).split('_')[-1].split('.')[0]
        file_date = datetime.strptime(file_date_str, '%Y%m%d')
        days_old = (datetime.now() - file_date).days

        return days_old > days
    except Exception as e:
        print(f"Error checking file age: {e}")
        return False

def initialize_data(create_model: bool = False) -> tuple[pl.DataFrame, Dict]:
    latest_bgg_file = get_latest_file(BGG_PATH)
    if is_file_older_than_days(latest_bgg_file):
        if input("The current data is older than 7 days, do you want to process new one? (y/n): ").lower() == 'y':
            create_model = True

    if is_data_folder_empty(step='bgg'):
        if input("BGG data folder is empty. Would you like to import data via BGG API and process it? (y/n): ").lower() == 'y':
            create_model = True
            bgg_df = asyncio.run(main_bgg(force_restart=True))
            games_data = run_data_preparation(bgg_df)
                
    elif is_data_folder_empty(step='prep'):
        answer = input("Processed data folder is empty. Would you like to: \n 1. Reprocess everything from the start (BGG > Data Prep). \n 2. Reprocess last fetched data from BGG? (1/2): ")
        if answer == '1':
            bgg_df = asyncio.run(main_bgg(force_restart=True))
            print('Processing BGG data...')
            games_data = run_data_preparation(bgg_df)
            create_model = True
        elif answer == '2':
            bgg_df = pl.read_parquet(get_latest_file(BGG_PATH))
            print('Processing BGG data...')
            games_data = run_data_preparation(bgg_df)
            create_model = True
            
    elif is_data_folder_empty(step='model'):
        if not create_model and input("There is no model saved. Do you want to create a new one? (y/n): ").lower() == 'y':
            create_model = True
            games_data = pl.read_parquet(get_latest_file(PREP_PATH))
            
    if create_model:
        if not 'games_data' in locals():
            bgg_df = asyncio.run(main_bgg(force_restart=True))
            games_data = run_data_preparation(bgg_df)
    else:
        games_data = pl.read_parquet(get_latest_file(PREP_PATH))

    model = bgClusters()
    clusters = model.fit(
        games_data,
        constraint_columns=CONSTRAINT_COLUMNS,
        name_column=NAME_COLUMN,
        restart_model=create_model,
        plot=True
    ).clusters

    return games_data, clusters

if not os.path.exists(FEEDBACK_DIR):
    os.makedirs(FEEDBACK_DIR)

if not os.path.exists(FEEDBACK_FILE):
    pl.DataFrame(FEEDBACK_STRUCTURE).write_csv(FEEDBACK_FILE)

# Flask routes
@app.route('/')
def index():
    global df
    game_names = df["game_name"].to_list()
    return render_template('index.html', game_names=game_names)

@app.route('/recommend', methods=['POST'])
def recommend():
    global rec
    game_name = request.form['game_name']
    try:
        recommendations = rec.recommend_games(game_name)
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
        feedback_entry = {
            'timestamp': [feedback_data['timestamp']],
            'input_game': [feedback_data['input_game']],
            'recommended_game': [feedback_data['recommended_game']],
            'feedback': [feedback_data['feedback']],
            'reason': [feedback_data.get('reason', None)],
            'comment': [feedback_data.get('comment', None)]
        }
        
        new_feedback_df = pl.DataFrame(feedback_entry)
        
        try:
            existing_df = pl.read_csv(FEEDBACK_FILE)
            for col in new_feedback_df.columns:
                if col not in existing_df.columns:
                    existing_df = existing_df.with_columns(pl.lit(None).alias(col))
            combined_df = pl.concat([existing_df, new_feedback_df], how="vertical")
        except Exception as e:
            print(f"Creating new feedback file: {str(e)}")
            combined_df = new_feedback_df
        
        combined_df.write_csv(FEEDBACK_FILE)

        return jsonify({"status": "success", "message": "Feedback saved successfully"})
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Error saving feedback: {str(e)}"
        }), 500
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BGG recommendation system')
    parser.add_argument('--force_restart', type=bool, default=False,
                       help='Force restart the model initialization')
    args = parser.parse_args()

    print("Initializing data and model...")
    df, clusters = initialize_data(create_model=args.force_restart)
    rec = RecommendationEngine(df, clusters)
    print("Starting Flask server...")
    app.run(debug=False, host='0.0.0.0', port=5000)