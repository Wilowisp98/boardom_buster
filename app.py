# app.py
from flask import Flask, render_template, request, jsonify
import polars as pl
import data_prep as data_prep
from model.binning_input_games import bin_board_games
from model.cluster_games import bgClusters
from model.recommend_games import RecommendationEngine
import os

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
    pl.DataFrame({
        'timestamp': [],
        'input_game': [],
        'recommended_game': [],
        'feedback': [],
        'reason': [],
        'comment': []
    }).write_csv(FEEDBACK_FILE)

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
        
        # Create base feedback entry with required fields
        feedback_entry = {
            'timestamp': [feedback_data['timestamp']],
            'input_game': [feedback_data['input_game']],
            'recommended_game': [feedback_data['recommended_game']],
            'feedback': [feedback_data['feedback']],
            'reason': [feedback_data.get('reason', None)],
            'comment': [feedback_data.get('comment', None)]
        }
        
        # Create DataFrame from the new feedback
        new_feedback_df = pl.DataFrame(feedback_entry)
        
        try:
            # Try to read existing feedback
            existing_df = pl.read_csv(FEEDBACK_FILE)
            # Ensure all columns exist in existing DataFrame
            for col in new_feedback_df.columns:
                if col not in existing_df.columns:
                    existing_df = existing_df.with_columns(pl.lit(None).alias(col))
            
            # Combine existing and new feedback
            combined_df = pl.concat([existing_df, new_feedback_df], how="vertical")
            
        except Exception as e:
            print(f"Creating new feedback file: {str(e)}")
            combined_df = new_feedback_df
        
        # Write the combined DataFrame back to CSV
        combined_df.write_csv(FEEDBACK_FILE)
        
        return jsonify({
            "status": "success", 
            "message": "Feedback saved successfully"
        })
        
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Error saving feedback: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)