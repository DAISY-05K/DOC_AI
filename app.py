from flask import Flask,session, render_template, request, jsonify, session
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
training = pd.read_csv("Training.csv")
testing = pd.read_csv("Testing.csv")
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
    with open("symptom_Description.csv") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) < 2:
                continue  # skip empty/malformed lines
            description_list[row[0].strip()] = row[1].strip()

def getSeverityDict():
    with open("Symptom_severity.csv") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) < 2:
                continue
            try:
                severityDictionary[row[0].strip()] = int(row[1])
            except ValueError:
                pass

def getprecautionDict():
    with open("symptom_precaution.csv") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) < 5:
                continue
            precautionDictionary[row[0].strip()] = [row[1], row[2], row[3], row[4]]


getSeverityDict(); getDescription(); getprecautionDict()

symptom_synonyms = {
    "stomach ache": "stomach_pain","belly pain": "stomach_pain","tummy pain": "stomach_pain","abdominal pain": "stomach_pain","gas pain": "stomach_pain",
    "cramps": "stomach_pain","stomach cramps": "stomach_pain","bloating": "stomach_pain","nausea": "nausea","vomiting": "vomiting","throwing up": "vomiting",
    "feeling sick": "nausea","diarrhea": "diarrhea","loose motion": "diarrhea","motions": "diarrhea","loose stool": "diarrhea","constipation": "constipation",
    "difficulty passing stool": "constipation","heartburn": "acid_reflux","acid reflux": "acid_reflux","indigestion": "acid_reflux",
    "high temperature": "fever","temperature": "fever","feaver": "fever","feverish": "fever","chills": "chills","cold": "chills","flu": "influenza","influenza": "influenza",
    "cough": "cough","coughing": "cough","dry cough": "cough","wet cough": "cough","throat pain": "sore_throat","sore throat": "sore_throat","throat irritation": "sore_throat",
    "breathing issue": "breathlessness","shortness of breath": "breathlessness","difficulty breathing": "breathlessness","wheezing": "breathlessness","runny nose": "runny_nose",
    "stuffy nose": "nasal_congestion","blocked nose": "nasal_congestion","sneezing": "sneezing","sinus pain": "sinusitis","sinus pressure": "sinusitis","nose bleed": "nosebleed",
    "body ache": "muscle_pain","body pain": "muscle_pain","muscle ache": "muscle_pain","joint pain": "joint_pain","leg pain": "joint_pain", "arm pain": "joint_pain",    "back pain": "back_pain","lower back pain": "back_pain",
    "neck pain": "neck_pain","fatigue": "fatigue","tiredness": "fatigue","weakness": "fatigue","diziness": "dizziness","lightheaded": "dizziness","fainting": "fainting",
    "loss of consciousness": "fainting","sweating": "sweating","shivering": "chills","headache": "headache","migraine": "headache","head pain": "headache","eye pain": "eye_pain",
    "eye strain": "eye_pain","blurred vision": "blurred_vision","double vision": "blurred_vision","drowsiness": "drowsiness","confusion": "confusion","memory loss": "memory_loss",
    "rash": "skin_rash","itching": "itching","skin irritation": "itching","hives": "skin_rash","acne": "acne","pimples": "acne","swelling": "swelling","bruising": "bruising",
    "bleeding": "bleeding","cuts": "wound","wound": "wound","burn": "burn","sunburn": "burn","frequent urination": "urinary_frequency","burning urination": "urinary_pain",
    "painful urination": "urinary_pain","blood in urine": "hematuria","menstrual pain": "menstrual_cramps","period cramps": "menstrual_cramps","missed period": "amenorrhea",
    "vaginal discharge": "vaginal_discharge","loss of appetite": "loss_of_appetite","poor appetite": "loss_of_appetite","weight loss": "weight_loss","weight gain": "weight_gain",
    "anxiety": "anxiety","stress": "stress","insomnia": "insomnia","trouble sleeping": "insomnia","depression": "depression"
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
    "🌺 Remember, self-care is not selfish.",
    "🍎 Take care of your body — it’s the only place you have to live.",
    "🧘‍♀️ Balance is not something you find, it’s something you create.",
    "🚶‍♂️ Small steps every day lead to big changes over time.",
    "🌻 Nourish your body, calm your mind, and feed your soul.",
    "💧 Drink water, stay positive, and keep moving forward."
]

# ------------------ State Machine ------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    # Render the about page (create templates/about.html if missing)
    return render_template('about.html')

@app.route('/support')
def support():
    # Support template file is `Support.html` in the templates directory.
    # Use the exact filename to avoid TemplateNotFound on case-sensitive filesystems.
    return render_template('Support.html')

@app.route('/index')
def index():
    session.clear()
    session['step'] = 'welcome'
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'message' not in data:
            return jsonify(reply="Invalid request: expected JSON {message: '...'}"), 400
        user_msg = data['message']
    except Exception as e:
        app.logger.exception('Error parsing request JSON')
        return jsonify(reply='Server error parsing request'), 500
    step = session.get('step', 'welcome')

    # replicate each console step
    
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening','morning','afternoon','evening']
    if step == 'welcome' or any(greet in user_msg for greet in greetings):
        session['step'] = 'name'
        return jsonify(reply="👋 Hello! I’m DR_AI, your healthcare assistant.\n✨ Welcome to the HealthCare ChatBot!\n➡️ What is your name?")

    elif step == 'name':
        session['name'] = user_msg
        session['step'] = 'age'
        return jsonify(reply=f"🥼 Please {session['name']} enter your age:")
    elif step == 'age':
        try:
            age = int(user_msg)
            if age <= 0 or age > 120:
                return jsonify(reply="⚠️ Please enter a valid age between 1 and 120:")
            session['age'] = age
        except ValueError:
            return jsonify(reply="⚠️ Age must be a number, please try again.")
        
        session['step'] = 'gender'
        return jsonify(reply=f"🥼 What is your gender, {session['name']}?\n (Male/Female/Other):")
    elif step == 'gender':
        try:
            gender = user_msg.strip().lower()
            if gender not in ['male', 'female', 'other', 'm', 'f', 'o']:
                return jsonify(reply="⚠️ Please enter Male, Female, or Other:")
            session['gender'] = gender
        except ValueError:
            return jsonify(reply="⚠️ Invalid input, please try again.")
        session['step'] = 'symptoms'
        return jsonify(reply=f"🥼 Please {session['name']} Describe your symptoms in a sentence, :")
    elif step == 'symptoms':
        symptoms_list = extract_symptoms(user_msg, cols)
        if not symptoms_list:
            return jsonify(reply="Sorry the symptoms you entered are beyond my capabilities. Please describe again:")
        session['symptoms'] = symptoms_list
        disease, conf, _ = predict_disease(symptoms_list)
        session['pred_disease'] = disease
        session['step'] = 'days'
        return jsonify(reply=f"✅ Detected symptoms: {', '.join(symptoms_list)}\n👉 For how many days have you had these symptoms?")
    elif step == 'days':
        session['days'] = user_msg
        session['step'] = 'severity'
        return jsonify(reply=f"🥼 Sorry, to hear about that {session['name']}, on a scale of 1–10, how severe is your condition?")
    elif step == 'severity':
        session['severity'] = user_msg
        session['step'] = 'preexist'
        return jsonify(reply="🥼 Do you have any pre-existing conditions ?")
    elif step == 'preexist':
        session['preexist'] = user_msg
        session['step'] = 'lifestyle'
        return jsonify(reply="🥼 Do you smoke, drink alcohol, or have irregular sleep?")
    elif step == 'lifestyle':
        session['lifestyle'] = user_msg
        session['step'] = 'family'
        return jsonify(reply="🥼 Any family history of similar illness?")
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

    # If no branch matched, return an informative message
    return jsonify(reply="I'm sorry, I couldn't process that. Please try again."), 400

def ask_next_symptom():
    i = session['ask_index']
    ds = session['disease_syms']
    if i < min(8, len(ds)):
        sym = ds[i]
        session['ask_index'] += 1
        return jsonify(reply=f"🥼 Do you have severe {sym.replace('_',' ')}? (yes/no):")
    else:
        session['step'] = 'final'
        return final_prediction()

def final_prediction():
    disease, conf, _ = predict_disease(session['symptoms'])
    about = description_list.get(disease, 'No description available.')
    precautions = precautionDictionary.get(disease, [])
    text = (f"                        Result                            \n"
            f"\n🩺 Based on the symptoms you you have provided you are likely suffering from, **{disease}**\n")
            
    if precautions:
        text += "\n\nThis are the suggested precautions:\n" + "\n\n".join(f"{i+1}. {p}" for i,p in enumerate(precautions))
    text += "\n\n\n💡 " + random.choice(quotes)
    text += f"\n\n\nThank you for using the chatbot. Wishing you good health, {session['name']}!"
    return jsonify(reply=text)
def process_message(self, user_input):
        """Process user message and return (response, end_of_chat: bool)"""
        user_input = user_input.strip().lower()

        # ---------------- SMALL TALK HANDLING ----------------
        # Only treat short messages as greetings/thanks/farewells
        farewells = ['bye', 'goodbye', 'see you', 'take care', 'farewell', 'later', 'see ya', 'catch you later', 'talk to you later', 'i am leaving', 'i have to go', 'got to go', 'see you later', 'bye bye', 'see ya later']
        thanks = ['thank you', 'thanks', 'thx', 'thank you very much']

        if len(user_input.split()) <= 3:
            if any(farewell in user_input for farewell in farewells):
                return "👋 Goodbye! Take care!", True
            if any(thank in user_input for thank in thanks):
                return "😊 You're welcome! If you have more questions, feel free to ask.", False

        # not a short small-talk message; continue normal processing
        return None, False

# ------------------ Run App ------------------
if __name__ == "__main__":
    app.run(debug=True)
