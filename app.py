from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import sqlite3
import pandas as pd
import pickle
import os
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import re
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime, timedelta

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
    # Table to store password reset tokens
    c.execute('''CREATE TABLE IF NOT EXISTS password_resets
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  token TEXT UNIQUE NOT NULL,
                  expires_at TEXT NOT NULL,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
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

# 3. Disease to Specialization Mapping (100+ diseases)
DISEASE_TO_SPECIALIZATION = {
    # Respiratory Diseases
    'flu': 'Pulmonologist',
    'common cold': 'General Physician',
    'asthma': 'Pulmonologist',
    'bronchitis': 'Pulmonologist',
    'pneumonia': 'Pulmonologist',
    'tuberculosis': 'Pulmonologist',
    'chronic obstructive pulmonary disease': 'Pulmonologist',
    'cough': 'General Physician',
    'persistent cough': 'Pulmonologist',
    'wheezing': 'Pulmonologist',
    'shortness of breath': 'Pulmonologist',
    
    # Cardiovascular Diseases
    'hypertension': 'Cardiologist',
    'heart disease': 'Cardiologist',
    'coronary artery disease': 'Cardiologist',
    'heart attack': 'Cardiologist',
    'arrhythmia': 'Cardiologist',
    'heart failure': 'Cardiologist',
    'angina': 'Cardiologist',
    'palpitations': 'Cardiologist',
    'high blood pressure': 'Cardiologist',
    'chest pain': 'Cardiologist',
    'stroke': 'Cardiologist',
    'atherosclerosis': 'Cardiologist',
    
    # Gastrointestinal Diseases
    'food poisoning': 'Gastroenterologist',
    'gastritis': 'Gastroenterologist',
    'ulcer': 'Gastroenterologist',
    'peptic ulcer': 'Gastroenterologist',
    'crohn disease': 'Gastroenterologist',
    'ibd': 'Gastroenterologist',
    'ibs': 'Gastroenterologist',
    'diarrhea': 'Gastroenterologist',
    'constipation': 'Gastroenterologist',
    'bloating': 'Gastroenterologist',
    'heartburn': 'Gastroenterologist',
    'acid reflux': 'Gastroenterologist',
    'gerd': 'Gastroenterologist',
    'appendicitis': 'Gastroenterologist',
    'hepatitis': 'Gastroenterologist',
    'liver cirrhosis': 'Gastroenterologist',
    'pancreatitis': 'Gastroenterologist',
    'stomach cancer': 'Gastroenterologist',
    
    # Endocrine Diseases
    'diabetes': 'Endocrinologist',
    'type 1 diabetes': 'Endocrinologist',
    'type 2 diabetes': 'Endocrinologist',
    'hyperthyroidism': 'Endocrinologist',
    'hypothyroidism': 'Endocrinologist',
    'thyroid disorder': 'Endocrinologist',
    'goiter': 'Endocrinologist',
    'hormone imbalance': 'Endocrinologist',
    'polycystic ovary syndrome': 'Endocrinologist',
    'pcos': 'Endocrinologist',
    'excessive thirst': 'Endocrinologist',
    'frequent urination': 'Endocrinologist',
    
    # Neurological Diseases
    'migraine': 'Neurologist',
    'headache': 'Neurologist',
    'epilepsy': 'Neurologist',
    'seizure': 'Neurologist',
    'alzheimers disease': 'Neurologist',
    'parkinsons disease': 'Neurologist',
    'multiple sclerosis': 'Neurologist',
    'cerebral palsy': 'Neurologist',
    'neuropathy': 'Neurologist',
    'vertigo': 'Neurologist',
    'dizziness': 'Neurologist',
    'stroke recovery': 'Neurologist',
    'motor neuron disease': 'Neurologist',
    'traumatic brain injury': 'Neurologist',
    
    # Mental Health
    'anxiety': 'Psychiatrist',
    'depression': 'Psychiatrist',
    'panic disorder': 'Psychiatrist',
    'bipolar disorder': 'Psychiatrist',
    'schizophrenia': 'Psychiatrist',
    'ptsd': 'Psychiatrist',
    'ocd': 'Psychiatrist',
    'stress disorder': 'Psychiatrist',
    'nervousness': 'Psychiatrist',
    'sleep disorder': 'Psychiatrist',
    'insomnia': 'Psychiatrist',
    
    # Orthopedic Diseases
    'fracture': 'Orthopedist',
    'bone fracture': 'Orthopedist',
    'osteoporosis': 'Orthopedist',
    'arthritis': 'Orthopedist',
    'osteoarthritis': 'Orthopedist',
    'rheumatoid arthritis': 'Orthopedist',
    'back pain': 'Orthopedist',
    'neck pain': 'Orthopedist',
    'joint pain': 'Orthopedist',
    'ligament injury': 'Orthopedist',
    'sports injury': 'Orthopedist',
    'spinal cord injury': 'Orthopedist',
    
    # Dermatological Diseases
    'acne': 'Dermatologist',
    'skin infection': 'Dermatologist',
    'eczema': 'Dermatologist',
    'psoriasis': 'Dermatologist',
    'dermatitis': 'Dermatologist',
    'urticaria': 'Dermatologist',
    'hives': 'Dermatologist',
    'fungal infection': 'Dermatologist',
    'ringworm': 'Dermatologist',
    'rash': 'Dermatologist',
    'skin rash': 'Dermatologist',
    'warts': 'Dermatologist',
    'skin cancer': 'Dermatologist',
    'melanoma': 'Dermatologist',
    'vitiligo': 'Dermatologist',
    
    # Ophthalmological Diseases
    'cataract': 'Ophthalmologist',
    'glaucoma': 'Ophthalmologist',
    'myopia': 'Ophthalmologist',
    'hyperopia': 'Ophthalmologist',
    'astigmatism': 'Ophthalmologist',
    'conjunctivitis': 'Ophthalmologist',
    'pink eye': 'Ophthalmologist',
    'retinopathy': 'Ophthalmologist',
    'diabetic retinopathy': 'Ophthalmologist',
    'blurred vision': 'Ophthalmologist',
    'eye infection': 'Ophthalmologist',
    
    # ENT Diseases
    'tonsillitis': 'ENT Specialist',
    'sore throat': 'ENT Specialist',
    'ear infection': 'ENT Specialist',
    'otitis media': 'ENT Specialist',
    'sinusitis': 'ENT Specialist',
    'rhinitis': 'ENT Specialist',
    'allergic rhinitis': 'ENT Specialist',
    'nasal congestion': 'ENT Specialist',
    'runny nose': 'ENT Specialist',
    'sneezing': 'ENT Specialist',
    'voice loss': 'ENT Specialist',
    'laryngitis': 'ENT Specialist',
    
    # Urological Diseases
    'kidney stones': 'Urologist',
    'urinary tract infection': 'Urologist',
    'uti': 'Urologist',
    'bladder infection': 'Urologist',
    'prostate disease': 'Urologist',
    'enlarged prostate': 'Urologist',
    'kidney disease': 'Urologist',
    'chronic kidney disease': 'Urologist',
    'kidney failure': 'Urologist',
    
    # Dental Diseases
    'tooth decay': 'Dentist',
    'cavity': 'Dentist',
    'dental caries': 'Dentist',
    'gingivitis': 'Dentist',
    'periodontitis': 'Dentist',
    'gum disease': 'Dentist',
    'root canal': 'Dentist',
    'tooth infection': 'Dentist',
    'tooth abscess': 'Dentist',
    
    # Obstetrics & Gynecology
    'pregnancy': 'Gynecologist',
    'pregnancy complications': 'Gynecologist',
    'gestational diabetes': 'Gynecologist',
    'preeclampsia': 'Gynecologist',
    'miscarriage': 'Gynecologist',
    'pelvic inflammatory disease': 'Gynecologist',
    'endometriosis': 'Gynecologist',
    'fibroids': 'Gynecologist',
    'menopause': 'Gynecologist',
    'irregular periods': 'Gynecologist',
    
    # Oncology
    'cancer': 'Oncologist',
    'lung cancer': 'Oncologist',
    'breast cancer': 'Oncologist',
    'colon cancer': 'Oncologist',
    'prostate cancer': 'Oncologist',
    'liver cancer': 'Oncologist',
    'ovarian cancer': 'Oncologist',
    'pancreatic cancer': 'Oncologist',
    'leukemia': 'Oncologist',
    'lymphoma': 'Oncologist',
    'tumor': 'Oncologist',
    'malignant tumor': 'Oncologist',
    
    # Infectious Diseases
    'covid-19': 'Infectious Disease Specialist',
    'coronavirus': 'Infectious Disease Specialist',
    'malaria': 'Infectious Disease Specialist',
    'dengue': 'Infectious Disease Specialist',
    'typhoid': 'Infectious Disease Specialist',
    'measles': 'Infectious Disease Specialist',
    'chickenpox': 'Infectious Disease Specialist',
    'whooping cough': 'Infectious Disease Specialist',
    'meningitis': 'Infectious Disease Specialist',
    'hiv aids': 'Infectious Disease Specialist',
    
    # General Conditions
    'fever': 'General Physician',
    'high fever': 'General Physician',
    'cold': 'General Physician',
    'fatigue': 'General Physician',
    'weakness': 'General Physician',
    'body ache': 'General Physician',
    'muscle pain': 'General Physician',
    'chills': 'General Physician',
    'weight loss': 'General Physician',
    'unexplained weight loss': 'General Physician',
    'nausea': 'General Physician',
    'vomiting': 'General Physician',
    'sweating': 'General Physician',
    'sensitivity to light': 'General Physician',
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
        app.logger.warning(f"REGISTER EMAIL RECEIVED: [{email}]")
        password = request.form.get('password')
        # Basic server-side validation
        if not name or not email or not password:
            flash("Please provide name, email, and password.", "danger")
            return redirect(url_for('register'))

        # simple email format check
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Please provide a valid email address.", "danger")
            return redirect(url_for('register'))

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password)

        try:
            app.logger.info(f"Register attempt for email: {email}")
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            # Check for existing email first to give a clear message
            c.execute("SELECT id FROM users WHERE email=?", (email,))
            existing = c.fetchone()
            app.logger.info(f"Existing check result: {existing}")
            if existing:
                conn.close()
                flash("Email is already registered. Please login or use a different email.", "warning")
                return redirect(url_for('login'))

            c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, hashed_pw))
            user_id = c.lastrowid
            app.logger.info(f"Inserted user id: {user_id}")
            conn.commit()
            conn.close()

            # Auto-login the newly registered user and redirect to dashboard
            session['user_id'] = user_id
            session['user_name'] = name
            flash("Registration successful — welcome!", "success")
            return redirect(url_for('home'))
        except Exception as e:
            # Log the exception for debugging
            app.logger.exception("Error during registration")
            flash("An unexpected error occurred during registration. Please try again.", "danger")
            return redirect(url_for('register'))
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # If already logged in, send to home/dashboard
    if 'user_id' in session:
        return redirect(url_for('home'))

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
    return redirect(url_for('login'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        app.logger.warning(f"[FORGOT] Request for email: [{email}]")
        app.logger.warning(f"[FORGOT] Current working dir: {os.getcwd()}")
        app.logger.warning(f"[FORGOT] DB file exists: {os.path.exists('users.db')}")
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=?", (email,))
        user = c.fetchone()
        conn.close()
        app.logger.warning(f"[FORGOT] User lookup result: {user}")
        # Better flow: generate a secure reset token and email a reset link.
        import secrets
        token = secrets.token_urlsafe(48)
        expires_at = (datetime.utcnow() + timedelta(hours=1)).isoformat()

        if user:
            user_id = user[0]
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            # remove any existing tokens for user
            c.execute("DELETE FROM password_resets WHERE user_id=?", (user_id,))
            # store raw token for link-based reset
            c.execute("INSERT INTO password_resets (user_id, token, expires_at) VALUES (?, ?, ?)", (user_id, token, expires_at))
            conn.commit()
            conn.close()

            reset_link = url_for('reset_password', token=token, _external=True)

            smtp_server = os.environ.get('SMTP_SERVER')
            smtp_port = int(os.environ.get('SMTP_PORT', 587))
            smtp_user = os.environ.get('SMTP_USERNAME')
            smtp_pass = os.environ.get('SMTP_PASSWORD')
            smtp_from = os.environ.get('SMTP_FROM', smtp_user)

            # Try to send reset link via email; if SMTP not configured or send fails, show link on page as fallback
            if smtp_server and smtp_user and smtp_pass:
                try:
                    msg = EmailMessage()
                    msg['Subject'] = 'SmartHealth AI — Password Reset'
                    msg['From'] = smtp_from
                    msg['To'] = email
                    msg.set_content(f"Hello,\n\nWe received a request to reset your SmartHealth AI password.\n\nClick the link below to set a new password (valid for 1 hour):\n\n{reset_link}\n\nIf you did not request this, ignore this message.\n\n— SmartHealth AI Team")

                    context = ssl.create_default_context()
                    with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                        server.starttls(context=context)
                        server.login(smtp_user, smtp_pass)
                        server.send_message(msg)

                    # render a confirmation message (masked email shown in template)
                    return render_template('forgot_password.html', email=email, reset_sent=True)
                except Exception:
                    app.logger.exception('SMTP send failed')
                    # fallback: show reset link on page
                    return render_template('forgot_password.html', reset_link=reset_link)
            else:
                # SMTP not configured — show reset link on page (dev fallback)
                return render_template('forgot_password.html', reset_link=reset_link)

        # generic fallback for non-existing emails
        app.logger.warning(f"[FORGOT] Email not found or hidden: [{email}]")
        flash("If this email exists, a password reset link has been sent. Check your email.", "success")
        return render_template('forgot_password.html', email=email, reset_sent=True)
    
    return render_template('forgot_password.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user_name=session.get('user_name'))


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    # validate token
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT user_id, expires_at FROM password_resets WHERE token=?", (token,))
    row = c.fetchone()
    conn.close()

    if not row:
        flash('Invalid or expired reset link.', 'danger')
        return redirect(url_for('login'))

    user_id, expires_at = row
    expires_dt = datetime.fromisoformat(expires_at)
    if datetime.utcnow() > expires_dt:
        # token expired
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("DELETE FROM password_resets WHERE token=?", (token,))
        conn.commit()
        conn.close()
        flash('Reset link has expired. Please request a new one.', 'danger')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        pw = request.form.get('password')
        confirm = request.form.get('confirm_password')
        if not pw or len(pw) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('reset_password.html', token=token)
        if pw != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('reset_password.html', token=token)

        hashed = generate_password_hash(pw)
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("UPDATE users SET password=? WHERE id=?", (hashed, user_id))
        c.execute("DELETE FROM password_resets WHERE token=?", (token,))
        conn.commit()
        conn.close()

        flash('Password has been changed. Please login with your new password.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)


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
        doctors_list = []
        other_suggestions = []
        
        if selected_location and selected_location != 'All':
            # Separate nearby and others
            nearby = filtered[filtered['location'] == selected_location]
            others = filtered[filtered['location'] != selected_location]
            
            if not nearby.empty:
                nearby = nearby.sort_values(by=['rating', 'experience'], ascending=False)
                doctors_list = nearby.head(10).to_dict(orient='records')
                if not others.empty:
                    others = others.sort_values(by=['rating', 'experience'], ascending=False)
                    other_suggestions = others.head(10).to_dict(orient='records')
            else:
                # No nearby, show all as others
                if not filtered.empty:
                    filtered = filtered.sort_values(by=['rating', 'experience'], ascending=False)
                    other_suggestions = filtered.head(10).to_dict(orient='records')
        else:
            # All locations
            if not filtered.empty:
                filtered = filtered.sort_values(by=['rating', 'experience'], ascending=False)
                doctors_list = filtered.head(10).to_dict(orient='records')

        # If no specialists found, fallback to General Physician
        if not doctors_list and not other_suggestions:
            gp_filtered = doctors_df[doctors_df['specialization'] == 'General Physician']
            if selected_location and selected_location != 'All':
                nearby_gp = gp_filtered[gp_filtered['location'] == selected_location]
                others_gp = gp_filtered[gp_filtered['location'] != selected_location]
                if not nearby_gp.empty:
                    nearby_gp = nearby_gp.sort_values(by=['rating', 'experience'], ascending=False)
                    doctors_list = nearby_gp.head(10).to_dict(orient='records')
                    if not others_gp.empty:
                        others_gp = others_gp.sort_values(by=['rating', 'experience'], ascending=False)
                        other_suggestions = others_gp.head(10).to_dict(orient='records')
                else:
                    if not gp_filtered.empty:
                        gp_filtered = gp_filtered.sort_values(by=['rating', 'experience'], ascending=False)
                        other_suggestions = gp_filtered.head(10).to_dict(orient='records')
            else:
                if not gp_filtered.empty:
                    gp_filtered = gp_filtered.sort_values(by=['rating', 'experience'], ascending=False)
                    doctors_list = gp_filtered.head(10).to_dict(orient='records')

        return jsonify({
            'status': 'success',
            'disease': prediction,
            'specialization': specialization,
            'doctors': doctors_list,
            'other_suggestions': other_suggestions,
            'searched_location': selected_location
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

if __name__ == '__main__':
    app.run(debug=True)