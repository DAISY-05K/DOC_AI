
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import re, random, pandas as pd, numpy as np, csv, warnings
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches

warnings.filterwarnings("ignore", category=DeprecationWarning)
app = Flask(__name__)
app.secret_key = "supersecret"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# ------------------ Load Data (unchanged) ------------------
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')
training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns  = testing.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]
testing  = testing.loc[:, ~testing.columns.duplicated()]
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

# Dictionaries
severityDictionary, description_list, precautionDictionary = {}, {}, {}
symptoms_dict = {symptom: idx for idx, symptom in enumerate(x)}

def getDescription():
    with open('symptom_Description.csv') as csv_file:
        for row in csv.reader(csv_file):
            description_list[row[0]] = row[1]

def getSeverityDict():
    with open('symptom_severity.csv') as csv_file:
        for row in csv.reader(csv_file):
            try: severityDictionary[row[0]] = int(row[1])
            except: pass

def getprecautionDict():
    with open('symptom_precaution.csv') as csv_file:
        for row in csv.reader(csv_file):
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

getSeverityDict(); getDescription(); getprecautionDict()

symptom_synonyms = {
    "stomach ache": "stomach_pain","belly pain": "stomach_pain","tummy pain": "stomach_pain","abdominal pain": "stomach_pain","gas pain": "stomach_pain",
    "cramps": "stomach_pain","stomach cramps": "stomach_pain","bloating": "stomach_pain","nausea": "nausea","vomiting": "vomiting","throwing up": "vomiting",
    "feeling sick": "nausea","diarrhea": "diarrhea","loose motion": "diarrhea","motions": "diarrhea","loose stool": "diarrhea","constipation": "constipation",
    "difficulty passing stool": "constipation","heartburn": "acid_reflux","acid reflux": "acid_reflux","indigestion": "acid_reflux","high temperature": "fever",
    "temperature": "fever","feaver": "fever","feverish": "fever","chills": "chills","cold": "chills","flu": "influenza","influenza": "influenza","cough": "cough",
    "coughing": "cough","dry cough": "cough","wet cough": "cough","throat pain": "sore_throat","sore throat": "sore_throat","throat irritation": "sore_throat",
    "breathing issue": "breathlessness","shortness of breath": "breathlessness","difficulty breathing": "breathlessness","wheezing": "breathlessness","runny nose": "runny_nose",
    "stuffy nose": "nasal_congestion","blocked nose": "nasal_congestion","sneezing": "sneezing","sinus pain": "sinusitis","sinus pressure": "sinusitis","nose bleed": "nosebleed",
    "body ache": "muscle_pain","body pain": "muscle_pain","muscle ache": "muscle_pain","joint pain": "joint_pain","leg pain": "joint_pain","arm pain": "joint_pain",
    "back pain": "back_pain","lower back pain": "back_pain","neck pain": "neck_pain","fatigue": "fatigue","tiredness": "fatigue","weakness": "fatigue","diziness": "dizziness",
    "lightheaded": "dizziness","fainting": "fainting","loss of consciousness": "fainting","sweating": "sweating","shivering": "chills","headache": "headache",
    "migraine": "headache","head pain": "headache","eye pain": "eye_pain","eye strain": "eye_pain","blurred vision": "blurred_vision","double vision": "blurred_vision",
    "drowsiness": "drowsiness","confusion": "confusion","memory loss": "memory_loss","rash": "skin_rash","itching": "itching","skin irritation": "itching","hives": "skin_rash",
    "acne": "acne","pimples": "acne","swelling": "swelling","bruising": "bruising","bleeding": "bleeding","cuts": "wound","wound": "wound","burn": "burn","sunburn": "burn",
    "frequent urination": "urinary_frequency","burning urination": "urinary_pain","painful urination": "urinary_pain","blood in urine": "hematuria","menstrual pain": "menstrual_cramps",
    "period cramps": "menstrual_cramps","missed period": "amenorrhea","vaginal discharge": "vaginal_discharge","loss of appetite": "loss_of_appetite","poor appetite": "loss_of_appetite",
    "weight loss": "weight_loss","weight gain": "weight_gain","anxiety": "anxiety","stress": "stress","insomnia": "insomnia","trouble sleeping": "insomnia","depression": "depression"
}

def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text: extracted.append(mapped)
    for symptom in all_symptoms:
        if symptom.replace("_"," ") in text: extracted.append(symptom)
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(word, [s.replace("_"," ") for s in all_symptoms], n=1, cutoff=0.8)
        if close:
            for sym in all_symptoms:
                if sym.replace("_"," ") == close[0]:
                    extracted.append(sym)
    return list(set(extracted))

def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class]*100,2)
    return disease, confidence, pred_proba

quotes = [
    "🌸 Health is wealth, take care of yourself.",
    "💪 A healthy outside starts from the inside.",
    "☀️ Every day is a chance to get stronger and healthier.",
    "🌿 Take a deep breath, your health matters the most.",
    "🌺 Remember, self-care is not selfish."
]

# ------------------ State Machine ------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/')
def index():
    session.clear()
    session['step'] = 'welcome'
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json['message']
    step = session.get('step', 'welcome')

    # replicate each console step
    if step == 'welcome':
        session['step'] = 'name'
        return jsonify(reply="✨ DOC_AI to HealthCare ChatBot!\n➡️ What is your name?")
    elif step == 'name':
        session['name'] = user_msg
        session['step'] = 'age'
        return jsonify(reply="➡️ Please enter your age:")
    elif step == 'age':
        session['age'] = user_msg
        session['step'] = 'gender'
        return jsonify(reply="➡️ What is your gender? (M/F/Other):")
    elif step == 'gender':
        session['gender'] = user_msg
        session['step'] = 'symptoms'
        return jsonify(reply="➡️ Describe your symptoms in a sentence:")
    elif step == 'symptoms':
        symptoms_list = extract_symptoms(user_msg, cols)
        if not symptoms_list:
            return jsonify(reply="❌ Could not detect valid symptoms. Please describe again:")
        session['symptoms'] = symptoms_list
        disease, conf, _ = predict_disease(symptoms_list)
        session['pred_disease'] = disease
        session['step'] = 'days'
        return jsonify(reply=f"✅ Detected symptoms: {', '.join(symptoms_list)}\n👉 For how many days have you had these symptoms?")
    elif step == 'days':
        session['days'] = user_msg
        session['step'] = 'severity'
        return jsonify(reply="➡️ On a scale of 1–10, how severe is your condition?")
    elif step == 'severity':
        session['severity'] = user_msg
        session['step'] = 'preexist'
        return jsonify(reply="➡️ Do you have any pre-existing conditions?")
    elif step == 'preexist':
        session['preexist'] = user_msg
        session['step'] = 'lifestyle'
        return jsonify(reply="➡️ Do you smoke, drink alcohol, or have irregular sleep?")
    elif step == 'lifestyle':
        session['lifestyle'] = user_msg
        session['step'] = 'family'
        return jsonify(reply="➡️ Any family history of similar illness?")
    elif step == 'family':
        session['family'] = user_msg
        # guided disease-specific questions
        disease = session['pred_disease']
        disease_symptoms = list(training[training['prognosis'] == disease].iloc[0][:-1].index[
            training[training['prognosis'] == disease].iloc[0][:-1] == 1
        ])
        session['disease_syms'] = disease_symptoms
        session['ask_index'] = 0
        session['step'] = 'guided'
        return ask_next_symptom()
    elif step == 'guided':
        # record yes/no
        idx = session['ask_index'] - 1
        if idx >= 0 and idx < len(session['disease_syms']):
            if user_msg.strip().lower() == 'yes':
                session['symptoms'].append(session['disease_syms'][idx])
        return ask_next_symptom()
    elif step == 'final':
        # already answered all guided
        return final_prediction()

def ask_next_symptom():
    i = session['ask_index']
    ds = session['disease_syms']
    if i < min(8, len(ds)):
        sym = ds[i]
        session['ask_index'] += 1
        return jsonify(reply=f"➡️ Do you also have {sym.replace('_',' ')}? (yes/no):")
    else:
        session['step'] = 'final'
        return final_prediction()

def final_prediction():
    disease, conf, _ = predict_disease(session['symptoms'])
    about = description_list.get(disease, 'No description available.')
    precautions = precautionDictionary.get(disease, [])
    text = (f"                        Result                            \n"
            f"\n🩺 Based on your answers, you may have **{disease}**\n"
            f"\n🔎 Confidence: {conf}%\n📖 About: {about}\n")
    if precautions:
        text += "\n\n🛡️ Suggested precautions:\n" + "\n\n".join(f"{i+1}. {p}" for i,p in enumerate(precautions))
    text += "\n\n\n💡 " + random.choice(quotes)
    text += f"\n\n\nThank you for using the chatbot. Wishing you good health, {session['name']}!"
    return jsonify(reply=text)
