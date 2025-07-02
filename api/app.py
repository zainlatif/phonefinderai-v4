from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
clf = joblib.load('../model/model.pkl')
vectorizer = joblib.load('../model/vectorizer.pkl')

# Load phone specs
phone_specs = pd.read_csv('../data/phone_specs.csv', encoding='utf-8-sig')
phone_specs.columns = phone_specs.columns.str.strip()
phone_specs['category'] = phone_specs['category'].str.strip()  # Add this line
phone_specs['category'] = phone_specs['category'].str.lower()
print(phone_specs.columns)  # Optional: Debug print

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    query_vec = vectorizer.transform([query])
    category = clf.predict(query_vec)[0].lower()
    print("Predicted category:", category)
    print("Available categories:", phone_specs['category'].unique())

    recommendations = phone_specs[phone_specs['category'] == category]['name'].tolist()

    return jsonify({'category': category, 'recommendations': recommendations})

if __name__ == '__main__':
    app.run(port=8000)
