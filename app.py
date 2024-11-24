# app.py
from flask import Flask, request, jsonify, render_template
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import json

app = Flask(__name__)

class SpamDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=1000,
            stop_words='english'
        )
        self.classifier = MultinomialNB()
        
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
        
    def extract_features(self, text):
        features = {
            'has_urgent': bool(any(word in text.lower() for word in ['urgent', 'immediate', 'action required'])),
            'has_money': bool(any(word in text.lower() for word in ['$', 'money', 'cash', 'prize', 'winner'])),
            'has_suspicious': bool(any(word in text.lower() for word in ['password', 'account', 'bank', 'verify'])),
            'excessive_caps': bool(len(re.findall(r'[A-Z]{3,}', text)) > 2),
            'multiple_exclamation': bool('!!' in text),
        }
        return features

    def train(self, X_train, y_train):
        X_train_processed = [self.preprocess_text(text) for text in X_train]
        X_train_tfidf = self.vectorizer.fit_transform(X_train_processed)
        self.classifier.fit(X_train_tfidf, y_train)
        
    def predict(self, text):
        processed_text = self.preprocess_text(text)
        text_tfidf = self.vectorizer.transform([processed_text])
        spam_prob = float(self.classifier.predict_proba(text_tfidf)[0][1])  # Convert to float
        
        features = self.extract_features(text)
        if sum(1 for v in features.values() if v) >= 3:  # Count True values
            spam_prob += 0.2
        
        spam_prob = min(max(spam_prob, 0), 1)
        
        return {
            'is_spam': bool(spam_prob > 0.5),  # Convert to bool explicitly
            'spam_probability': round(float(spam_prob * 100), 2),  # Convert to float
            'suspicious_features': [k for k, v in features.items() if v]
        }

# Initialize detector
detector = SpamDetector()

# Train with sample data
sample_emails = [
    "Hey, how are you doing?",
    "URGENT: You've won $1,000,000!!! Claim NOW!",
    "Meeting at 3pm tomorrow",
    "VERIFY YOUR ACCOUNT NOW! $500 FREE CASH!!!",
    "Hello there, just checking in",
    "FREE MONEY! CLICK NOW! URGENT!!!",
    "Can we meet for coffee?",
    "Project status update needed"
]
sample_labels = [0, 1, 0, 1, 0, 1, 0, 0]  # 0 for ham, 1 for spam
detector.train(sample_emails, sample_labels)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check_spam', methods=['POST'])
def check_spam():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
            
        result = detector.predict(message)
        # Ensure all values are JSON serializable
        return jsonify({
            'is_spam': bool(result['is_spam']),
            'spam_probability': float(result['spam_probability']),
            'suspicious_features': list(result['suspicious_features'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)