from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, pandas as pd, re

app = Flask(__name__)
CORS(app)

clf       = joblib.load('../model/model.pkl')
vectorizer= joblib.load('../model/vectorizer.pkl')

# Now camera_quality, battery_life, performance are numeric!
phone_specs = pd.read_csv('../data/phone_specs.csv', encoding='utf-8-sig')
phone_specs.columns = phone_specs.columns.str.strip()
phone_specs['category'] = phone_specs['category'].str.lower()
phone_specs['price']    = phone_specs['price'].astype(int)

@app.route('/predict', methods=['POST'])
def predict():
    query = request.get_json().get('query','').lower().strip()
    if not query:
        return jsonify({'error':'No query provided'}), 400

    cat = clf.predict(vectorizer.transform([query]))[0].lower()
    df  = phone_specs[phone_specs['category']==cat]

    # price cap?
    nums = re.findall(r'\d+', query)
    if nums:
        cap = int(nums[0])
        df = df[df['price'] <= cap]

    # pick score column
    if cat=='camera_phone':
        score_col='camera_quality'
    elif cat=='gaming_phone':
        score_col='performance'
    elif cat=='long_battery':
        score_col='battery_life'
    else:
        score_col='price'  # or leave as-is for budget/rugged

    # sort: best score desc, then cheapest
    df = df.sort_values(by=[score_col,'price'], ascending=[False,True])

    # return top N
    n = 3 if nums else 5
    recs = df.head(n).to_dict(orient='records')

    return jsonify({'category':cat,'recommendations':recs})

if __name__=='__main__':
    app.run(port=8000)
