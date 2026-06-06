from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load saved model
print("Loading model...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('aro_data.pkl', 'rb') as f:
    aro_data = pickle.load(f)
print("Model loaded!")

def extract_kmers(sequence, k=4):
    return ' '.join([sequence[i:i+k] for i in range(len(sequence)-k+1)])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    sequence = data.get('sequence', '').upper().strip()
    
    if len(sequence) < 100:
        return jsonify({'error': 'Sequence too short. Please enter at least 100 bases.'})
    
    kmers = extract_kmers(sequence)
    X = vectorizer.transform([kmers])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    if prediction == 1:
        matches = aro_data[aro_data['CARD Short Name'].notna()].head(3)
        resistance_info = []
        for _, row in matches.iterrows():
            resistance_info.append({
                'gene': row['CARD Short Name'],
                'drug_class': row['Drug Class'],
                'mechanism': row['Resistance Mechanism']
            })
        
        return jsonify({
            'result': 'resistant',
            'confidence': round(float(probability[1]) * 100, 2),
            'resistance_info': resistance_info
        })
    else:
        return jsonify({
            'result': 'sensitive',
            'confidence': round(float(probability[0]) * 100, 2),
            'resistance_info': []
        })

if __name__ == '__main__':
    app.run(debug=True)