import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import joblib

# Load dataset
df = pd.read_csv("news.csv")
df.dropna(inplace=True)

# Features and labels
X = df['text']
y = df['label']

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_vec, y)

# Save model and vectorizer
joblib.dump(model, 'model/news_classifier.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')

print("✅ Model and vectorizer saved to 'model/' folder.")
