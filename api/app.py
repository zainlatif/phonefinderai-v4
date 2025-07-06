from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import re

app = Flask(__name__)
CORS(app)

# Load trained model and vectorizer
clf = joblib.load('../model/model.pkl')
vectorizer = joblib.load('../model/vectorizer.pkl')

# Load phone specifications
phone_specs = pd.read_csv('../data/phone_specs.csv', encoding='utf-8-sig')
phone_specs.columns = phone_specs.columns.str.strip()
phone_specs['category'] = phone_specs['category'].str.strip().str.lower()
phone_specs['price'] = phone_specs['price'].astype(int)  # Ensure price is integer

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    query = data.get('query', '').lower().strip()

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Step 1: Predict category using ML model
    query_vec = vectorizer.transform([query])
    category = clf.predict(query_vec)[0].lower()

    # Step 2: Filter phones by category
    filtered_phones = phone_specs[phone_specs['category'] == category]

    # Step 3: Extract price from query using regex
    price_matches = re.findall(r'\d+', query)
    if price_matches:
        max_price = int(price_matches[0])
        filtered_phones = filtered_phones[filtered_phones['price'] <= max_price]

    # Step 4: Apply more specific filters based on keywords
    if "long battery" in query or "battery backup" in query or "good battery" in query:
        filtered_phones = filtered_phones[filtered_phones["battery_life"].isin(['excellent', 'good'])]

    if category == "camera_phone" and (
        "best camera" in query or "good camera" in query or "photography" in query
    ):
        filtered_phones = filtered_phones.sort_values(by="camera_quality", ascending=False)

    if "gaming" in query or "performance" in query or "fast phone" in query:
        filtered_phones = filtered_phones.sort_values(by="performance", ascending=False)

    # Step 5: Return top 3 recommendations
    if filtered_phones.empty:
        return jsonify({
            'category': category,
            'recommendations': [],
            'message': '❌ No matching phones found for your query.'
        })

    top_recommendations = filtered_phones.sort_values(by="price").head(3).to_dict(orient="records")

    return jsonify({
        'category': category,
        'recommendations': top_recommendations
    })

if __name__ == '__main__':
    app.run(port=8000)
