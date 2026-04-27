from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import sqlite3
import pandas as pd
import pickle
import os
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_secret_key")

# Stop words for simple NLP
STOP_WORDS = {'i', 'have', 'and', 'my', 'is', 'a', 'the', 'feeling', 'suffering', 'from', 'with', 'me', 'it', 'am', 'was', 'for', 'to', 'in'}

def clean_natural_input(text):
    # Remove punctuation and split into words
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    # Filter out stop words
    cleaned_words = [w for w in words if w not in STOP_WORDS]
    return " ".join(cleaned_words)

# 0. Database Setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  name TEXT NOT NULL, 
                  email TEXT UNIQUE NOT NULL, 
                  password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

# 1. Models Load Karne Ka Logic
tfidf = None
model = None
disease_model = None
symptom_list = None

try:
    disease_model_path = 'disease_model.pkl'
    symptom_list_path = 'symptom_list.pkl'

    if os.path.exists(disease_model_path) and os.path.exists(symptom_list_path):
        disease_model = pickle.load(open(disease_model_path, 'rb'))
        symptom_list = pickle.load(open(symptom_list_path, 'rb'))
        print("Disease Prediction Models Loaded Successfully")

except Exception as e:
    print(f"Error loading models: {e}.")

# 2. Dataset Load Karne Ka Logic
CSV_PATH = 'data/doctor_dataset.csv'
doctors_df = None
if os.path.exists(CSV_PATH):
    doctors_df = pd.read_csv(CSV_PATH)
    print("Dataset Loaded Successfully")
else:
    print("Error: doctor_dataset.csv not found!")

# 3. Disease to Specialization Mapping
DISEASE_TO_SPECIALIZATION = {
    'flu': 'General Physician',
    'common cold': 'General Physician',
    'asthma': 'Pulmonologist',
    'diabetes': 'Endocrinologist',
    'anxiety': 'Psychiatrist',
    'panic disorder': 'Psychiatrist',
    'cataract': 'Ophthalmologist',
    'acne': 'Dermatologist',
    'migraine': 'Neurologist',
    'hypertension': 'Cardiologist',
    'fracture': 'Orthopedic',
    'tonsillitis': 'ENT Specialist',
    'tooth decay': 'Dentist',
    'kidney stones': 'Urologist',
    'pregnancy': 'Gynecologist',
    'cancer': 'Oncologist',
    'lung': 'Pulmonologist',
    'food poisoning': 'Gastroenterologist',
    'cold': 'General Physician',
    'cough': 'General Physician',
    'skin infection': 'Dermatologist',
    'acne': 'Dermatologist',
    'fracture': 'Orthopedic',
    'stomach pain': 'Gastroenterologist',
    'heart': 'Cardiologist',
    'brain': 'Neurologist',
    'kidney': 'Urologist',
    'bone': 'Orthopedic',
    'teeth': 'Dentist',
    'skin': 'Dermatologist'
}

def get_specialist(disease_name):
    if not disease_name: return 'General Physician'
    d_lower = disease_name.lower()
    for key, val in DISEASE_TO_SPECIALIZATION.items():
        if key in d_lower:
            return val
    return 'General Physician'

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_name = session.get('user_name')
    return render_template('index.html', user_name=user_name)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        hashed_pw = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, hashed_pw))
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Email already registered or database error!", "danger")
            return redirect(url_for('register'))
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=?", (email,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            return redirect(url_for('home'))
        else:
            flash("Invalid email or password. Please try again.", "danger")
            return redirect(url_for('login'))
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user_name=session.get('user_name'))

@app.route('/api/locations', methods=['GET'])
def get_locations():
    try:
        if doctors_df is not None:
            locations = sorted(doctors_df['location'].unique().tolist())
            return jsonify({'locations': locations})
        return jsonify({'locations': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        raw_input = data.get('symptoms', '')
        user_input = clean_natural_input(raw_input)
        
        # --- Edge Case: Khali input ya bahut chota input ---
        if len(user_input) < 3:
            return jsonify({'error': "Please enter valid symptoms (at least 3 characters)."}), 400

        # 0. Check if user is searching for a specific doctor name
        name_search = doctors_df[doctors_df['name'].str.contains(user_input, case=False, na=False)]
        if not name_search.empty and len(user_input) > 5:
            selected_location = data.get('location', 'All')
            if selected_location != 'All':
                nearby = name_search[name_search['location'] == selected_location].to_dict(orient='records')
                others = name_search[name_search['location'] != selected_location].to_dict(orient='records')
                return jsonify({
                    'specialization': "Doctor Search",
                    'doctors': nearby,
                    'other_suggestions': others,
                    'searched_location': selected_location
                })
            return jsonify({
                'specialization': "Doctor Search",
                'doctors': name_search.to_dict(orient='records'),
                'searched_location': 'All'
            })

        # 1. Direct Keyword Match (Check if user entered a disease name directly)
        prediction = None
        specialization = None
        
        for disease, spec in DISEASE_TO_SPECIALIZATION.items():
            if disease in user_input:
                prediction = disease.capitalize()
                specialization = spec
                break
        
        # 2. AI Prediction (if no direct match)
        if not specialization:
            if disease_model is None or symptom_list is None:
                return jsonify({'error': "AI models are not loaded. Please contact administrator."}), 503

            # Create binary vector from user input
            input_vector = np.zeros(len(symptom_list))
            found_any = False
            for i, symptom in enumerate(symptom_list):
                # Flexible matching (replace underscores with spaces)
                if symptom.lower().replace('_', ' ') in user_input.lower():
                    input_vector[i] = 1
                    found_any = True
            
            if not found_any:
                return jsonify({'error': "No matching symptoms found. Please try different terms."}), 404
                
            # Predict Disease
            prediction = disease_model.predict([input_vector])[0]
            specialization = get_specialist(prediction)
        
        # 2. Database (CSV) Filter with Fallback Logic
        selected_location = data.get('location', 'All')
        
        # Try to get specialists
        filtered = doctors_df[doctors_df['specialization'] == specialization]
        final_doctors = pd.DataFrame()
        
        # Step 1: Specific Specialist in Specific Location
        if selected_location and selected_location != 'All':
            loc_filtered = filtered[filtered['location'] == selected_location]
            if not loc_filtered.empty:
                final_doctors = loc_filtered
        
        # Step 2: Fallback - Specific Specialist in Any Location
        if final_doctors.empty:
            final_doctors = filtered
        
        # Step 3: Fallback - General Physician in Specific Location
        if final_doctors.empty and specialization != 'General Physician':
            gp_filtered = doctors_df[doctors_df['specialization'] == 'General Physician']
            if selected_location and selected_location != 'All':
                loc_gp = gp_filtered[gp_filtered['location'] == selected_location]
                if not loc_gp.empty:
                    final_doctors = loc_gp
        
        # Step 4: Absolute Fallback - General Physician in Any Location
        if final_doctors.empty:
            final_doctors = doctors_df[doctors_df['specialization'] == 'General Physician']

        # Sort and limit
        if not final_doctors.empty:
            final_doctors = final_doctors.sort_values(by=['rating', 'experience'], ascending=False)
            
        doctors_list = final_doctors.head(5).to_dict(orient='records')

        return jsonify({
            'status': 'success',
            'disease': prediction,
            'specialization': specialization,
            'doctors': doctors_list,
            'searched_location': selected_location
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        if disease_model is None or symptom_list is None:
            return jsonify({'error': "Disease model not loaded."}), 503
            
        # Get symptoms from form or JSON
        if request.is_json:
            raw_input = request.json.get('symptoms', '')
        else:
            raw_input = request.form.get('symptoms', '')
            
        user_input = clean_natural_input(raw_input)
            
        if not user_input or user_input.strip() == "":
            return jsonify({'error': "Please enter symptoms"}), 400
            
        # Process comma-separated symptoms
        input_symptoms = [s.strip().lower() for s in user_input.split(',')]
        
        # Create binary vector
        input_vector = np.zeros(len(symptom_list))
        for i, symptom in enumerate(symptom_list):
            if symptom.lower().replace('_', ' ') in user_input.lower():
                input_vector[i] = 1

        # Check if any symptoms matched
        if input_vector.sum() == 0:
            return jsonify({'error': "No matching symptoms found. Please try different terms."}), 400
            
        # Predict
        prediction = disease_model.predict([input_vector])[0]
        
        # Recommendation Logic
        recommended_doctors = []
        specialization = get_specialist(prediction)
        
        if specialization and doctors_df is not None:
            # Try to get specialists
            filtered = doctors_df[doctors_df['specialization'] == specialization]
            
            # Filter by location if provided
            selected_location = request.args.get('location') or (request.json.get('location') if request.is_json else request.form.get('location'))
            
            final_doctors = pd.DataFrame()
            
            # Step 1: Specific Specialist in Specific Location
            if selected_location and selected_location != 'All':
                loc_filtered = filtered[filtered['location'] == selected_location]
                if not loc_filtered.empty:
                    final_doctors = loc_filtered
            
            # Step 2: Fallback - Specific Specialist in Any Location
            if final_doctors.empty:
                final_doctors = filtered
            
            # Step 3: Fallback - General Physician in Specific Location
            if final_doctors.empty and specialization != 'General Physician':
                gp_filtered = doctors_df[doctors_df['specialization'] == 'General Physician']
                if selected_location and selected_location != 'All':
                    loc_gp = gp_filtered[gp_filtered['location'] == selected_location]
                    if not loc_gp.empty:
                        final_doctors = loc_gp
            
            # Step 4: Absolute Fallback - General Physician in Any Location
            if final_doctors.empty:
                final_doctors = doctors_df[doctors_df['specialization'] == 'General Physician']

            # Sort and return
            if not final_doctors.empty:
                final_doctors = final_doctors.sort_values(by=['rating', 'experience'], ascending=False)
                recommended_doctors = final_doctors.head(5).to_dict(orient='records')
        
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'specialization': specialization,
            'doctors': recommended_doctors,
            'input_received': user_input
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)