from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

model = None
scaler = None
team_recent_wins = None
team_form_wins = None
h2h_recent_records = None
venue_recent_stats = None
recent_team_performance = None
venue_recent_features = None

def load_model_components():
    global model, scaler, team_recent_wins, team_form_wins, h2h_recent_records
    global venue_recent_stats, recent_team_performance, venue_recent_features
    
    try:
        # Load main components
        model = pickle.load(open('recent_model.pkl', 'rb'))
        scaler = pickle.load(open('recent_scaler.pkl', 'rb'))
        team_recent_wins = pickle.load(open('team_recent_wins.pkl', 'rb'))
        team_form_wins = pickle.load(open('team_form_wins.pkl', 'rb'))
        h2h_recent_records = pickle.load(open('h2h_recent_records.pkl', 'rb'))
        venue_recent_stats = pickle.load(open('venue_recent_stats.pkl', 'rb'))
        recent_team_performance = pickle.load(open('recent_team_performance.pkl', 'rb'))
        venue_recent_features = pickle.load(open('venue_recent_features.pkl', 'rb'))
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def predict_match_result(team1, team2, venue, toss_winner, toss_decision):
    # 1. Team strength from recent performance
    team1_recent_wins = team_recent_wins.get(team1, 0)
    team2_recent_wins = team_recent_wins.get(team2, 0)
    team1_form = team_form_wins.get(team1, 0)
    team2_form = team_form_wins.get(team2, 0)
    
    team1_strength = 0.6 * team1_form + 0.4 * team1_recent_wins
    team2_strength = 0.6 * team2_form + 0.4 * team2_recent_wins
    strength_difference = team1_strength - team2_strength
    
    # 2. H2H advantage
    teams = tuple(sorted([team1, team2]))
    if teams in h2h_recent_records:
        team1_h2h = h2h_recent_records[teams].get(team1, 0)
        team2_h2h = h2h_recent_records[teams].get(team2, 0)
        total_h2h = team1_h2h + team2_h2h
        h2h_advantage = (team1_h2h - team2_h2h) / max(total_h2h, 1) if total_h2h >= 3 else 0
    else:
        h2h_advantage = 0
    
    # 3. Venue advantage
    team1_venue_pair = (venue, team1)
    team2_venue_pair = (venue, team2)
    
    team1_venue_rate = 0.5
    team2_venue_rate = 0.5
    
    if team1_venue_pair in venue_recent_stats and venue_recent_stats[team1_venue_pair]['matches'] >= 3:
        team1_venue_rate = venue_recent_stats[team1_venue_pair]['wins'] / venue_recent_stats[team1_venue_pair]['matches']
        
    if team2_venue_pair in venue_recent_stats and venue_recent_stats[team2_venue_pair]['matches'] >= 3:
        team2_venue_rate = venue_recent_stats[team2_venue_pair]['wins'] / venue_recent_stats[team2_venue_pair]['matches']
    
    venue_advantage = team1_venue_rate - team2_venue_rate
    
    # 4. Performance trends
    team1_trend = recent_team_performance.get(team1, 0.5)
    team2_trend = recent_team_performance.get(team2, 0.5)
    performance_trend_diff = team1_trend - team2_trend
    
    # 5. Toss factors
    toss_advantage = 1 if toss_winner == team1 else 0
    bat_first = 1 if toss_decision == 'bat' else 0
    
    # 6. Venue characteristics
    venue_avg_score = venue_recent_features['avg_score'].get(venue, 160)
    venue_bat_first_avg = venue_recent_features['batting_first'].get(venue, 160)
    venue_chase_rate = venue_recent_features['chase_success'].get(venue, 0.5)
    venue_bat_preference = venue_recent_features['bat_preference'].get(venue, 0.5)
    venue_close_rate = venue_recent_features['close_rate'].get(venue, 0.5)
    
    toss_decision_smart = 1 if (toss_decision == 'bat' and venue_bat_preference > 0.6) or \
                              (toss_decision == 'field' and venue_bat_preference < 0.4) else 0
    
    current_form_diff = 0  # Simplified for prediction
    
    # Create feature array (15 features same as training)
    X_input = np.array([[
        team1_strength, team2_strength, strength_difference,
        h2h_advantage, venue_advantage, performance_trend_diff,
        toss_advantage, bat_first, toss_decision_smart,
        venue_avg_score, venue_chase_rate, venue_bat_preference,
        venue_close_rate, current_form_diff,
        (venue_bat_first_avg - venue_avg_score) / max(venue_avg_score, 1)
    ]])
    
    # Scale and predict
    X_input_scaled = scaler.transform(X_input)
    team1_win_probability = model.predict_proba(X_input_scaled)[0][1]
    team2_win_probability = 1 - team1_win_probability
    
    predicted_winner = team1 if team1_win_probability > team2_win_probability else team2
    confidence = max(team1_win_probability, team2_win_probability)
    
    return {
        'predicted_winner': predicted_winner,
        'team1_probability': round(team1_win_probability * 100, 2),
        'team2_probability': round(team2_win_probability * 100, 2),
        'confidence': round(confidence * 100, 2),
        'team1_recent_wins': team1_recent_wins,
        'team2_recent_wins': team2_recent_wins,
        'h2h_advantage': round(h2h_advantage, 3),
        'venue_advantage': round(venue_advantage, 3)
    }

@app.route('/')
def home():
    return jsonify({"message": "IPL Match Predictor API is running"})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        team1 = data['team1']
        team2 = data['team2']
        venue = data['venue']
        toss_winner = data['toss_winner']
        toss_decision = data['toss_decision']
        
        # Basic validation
        if team1 == team2:
            return jsonify({"error": "Please select different teams"}), 400
        
        if toss_winner not in [team1, team2]:
            return jsonify({"error": "Toss winner must be one of the playing teams"}), 400
        
        # Make prediction
        result = predict_match_result(team1, team2, venue, toss_winner, toss_decision)
        
        return jsonify({
            "success": True,
            "result": result,
            "match_data": {
                'team1': team1, 'team2': team2, 'venue': venue,
                'toss_winner': toss_winner, 'toss_decision': toss_decision
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Error making prediction: {str(e)}"}), 500

if __name__ == '__main__':
    if load_model_components():
        app.run(debug=True, host='0.0.0.0', port=5000)