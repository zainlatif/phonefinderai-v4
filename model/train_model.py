import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
df = pd.read_csv('../data/training_data.csv')
df['query'] = df['query'].astype(str)
df['label'] = df['label'].astype(str)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['query'])
y = df['label']

# Train model
clf = LogisticRegression(class_weight='balanced', max_iter=1000)
clf.fit(X, y)

# Save model and vectorizer
joblib.dump(clf, '../model/model.pkl')
joblib.dump(vectorizer, '../model/vectorizer.pkl')
