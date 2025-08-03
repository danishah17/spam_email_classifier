import pandas as pd
import numpy as np
import re
import warnings
import threading
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from collections import Counter
import pickle
import os
warnings.filterwarnings('ignore')

from flask import Flask, render_template_string, request, jsonify, session
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

class IndustrySpamClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.feature_selector = None
        self.stemmer = PorterStemmer() if NLTK_AVAILABLE else None
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        self.sentiment_analyzer = SentimentIntensityAnalyzer() if NLTK_AVAILABLE else None
        self.stop_words = set(stopwords.words('english')) if NLTK_AVAILABLE else self._get_basic_stopwords()
        self.trained = False
        self.spam_keywords = self._get_spam_keywords()
        self.model_version = "1.0"
        self.training_history = []
        
    def _get_basic_stopwords(self):
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
    
    def _get_spam_keywords(self):
       
        return {
            'money': ['free', 'cash', 'money', 'prize', 'win', 'winner', 'won', 'earn', 'profit', 'income', 'dollar', '$', '£', '€'],
            'urgency': ['urgent', 'immediately', 'now', 'asap', 'hurry', 'quick', 'fast', 'instant', 'limited time', 'expires'],
            'offers': ['offer', 'deal', 'discount', 'sale', 'promotion', 'bonus', 'gift', 'reward', 'guarantee'],
            'suspicious': ['click here', 'click now', 'call now', 'order now', 'buy now', 'act now', 'dont miss', 'limited offer'],
            'scam': ['congratulations', 'selected', 'chosen', 'lottery', 'jackpot', 'million', 'inheritance', 'beneficiary'],
            'phishing': ['verify', 'confirm', 'update', 'suspend', 'account', 'login', 'password', 'security', 'expires']
        }
    
    def extract_advanced_features(self, text):
       
        text_lower = str(text).lower()
        
        features = {
            # Basic features
            'length': len(text),
            'word_count': len(text.split()),
            'char_count': len(text),
            'sentence_count': len(sent_tokenize(text)) if NLTK_AVAILABLE else text.count('.') + text.count('!') + text.count('?'),
            
            # Character-based features
            'caps_ratio': len(re.findall(r'[A-Z]', text)) / max(len(text), 1),
            'digit_ratio': len(re.findall(r'\d', text)) / max(len(text), 1),
            'special_char_ratio': len(re.findall(r'[!@#$%^&*()_+=\[\]{}|;:,.<>?]', text)) / max(len(text), 1),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            
            # Spam keyword features
            'money_keywords': sum(1 for word in self.spam_keywords['money'] if word in text_lower),
            'urgency_keywords': sum(1 for word in self.spam_keywords['urgency'] if word in text_lower),
            'offer_keywords': sum(1 for word in self.spam_keywords['offers'] if word in text_lower),
            'suspicious_keywords': sum(1 for word in self.spam_keywords['suspicious'] if word in text_lower),
            'scam_keywords': sum(1 for word in self.spam_keywords['scam'] if word in text_lower),
            'phishing_keywords': sum(1 for word in self.spam_keywords['phishing'] if word in text_lower),
            
            # Pattern-based features
            'has_urls': len(re.findall(r'http[s]?://|www\.|\.com|\.org|\.net', text_lower)),
            'has_emails': len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
            'has_phone': len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\d{10}\b', text)),
            'has_currency': len(re.findall(r'[$£€¥₹]\d+|\d+\s*(dollar|pound|euro|rupee)', text_lower)),
            
            # Linguistic features
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'unique_words_ratio': len(set(text.lower().split())) / max(len(text.split()), 1),
        }
        
        # Sentiment analysis
        if self.sentiment_analyzer:
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            features.update({
                'sentiment_compound': sentiment['compound'],
                'sentiment_positive': sentiment['pos'],
                'sentiment_negative': sentiment['neg'],
                'sentiment_neutral': sentiment['neu']
            })
        
        return features
    
    def advanced_preprocess(self, text):
        """Advanced text preprocessing with multiple techniques"""
        text = str(text).lower()
        
        # Extract features before cleaning
        features = self.extract_advanced_features(text)
        
        # Clean text
        text = re.sub(r'http[s]?://\S+|www\.\S+', ' URL ', text)  # Replace URLs
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL ', text)  # Replace emails
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\d{10}\b', ' PHONE ', text)  # Replace phone numbers
        text = re.sub(r'[$£€¥₹]\d+|\d+\s*(dollar|pound|euro|rupee)', ' MONEY ', text)  # Replace money
        text = re.sub(r'\d+', ' NUMBER ', text)  # Replace numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        
        # Tokenize and process
        if NLTK_AVAILABLE and self.stemmer and self.lemmatizer:
            try:
                words = word_tokenize(text)
                # Apply lemmatization and stemming
                words = [self.lemmatizer.lemmatize(self.stemmer.stem(word)) 
                        for word in words if word not in self.stop_words and len(word) > 2]
            except:
                words = [word for word in text.split() if word not in self.stop_words and len(word) > 2]
        else:
            words = [word for word in text.split() if word not in self.stop_words and len(word) > 2]
        
        processed_text = ' '.join(words)
        
        # Add feature indicators based on extracted features
        if features['money_keywords'] > 0:
            processed_text += ' MONEY_INDICATOR'
        if features['urgency_keywords'] > 0:
            processed_text += ' URGENCY_INDICATOR'
        if features['caps_ratio'] > 0.3:
            processed_text += ' CAPS_INDICATOR'
        if features['has_urls'] > 0:
            processed_text += ' URL_INDICATOR'
        if features['has_phone'] > 0:
            processed_text += ' PHONE_INDICATOR'
        if features['exclamation_count'] > 2:
            processed_text += ' EXCITEMENT_INDICATOR'
        if features['scam_keywords'] > 0:
            processed_text += ' SCAM_INDICATOR'
        if features['phishing_keywords'] > 0:
            processed_text += ' PHISHING_INDICATOR'
            
        return processed_text
    
    def train_model(self):
        """Train an optimized ensemble model with hyperparameter tuning"""
        print(" Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv('spam.csv')
        X = df.iloc[:, 1].values
        y = np.where(df.iloc[:, 0].values == 'spam', 1, 0)
        
        print(f" Dataset loaded: {len(X)} samples ({sum(y)} spam, {len(y)-sum(y)} ham)")
        
        # Preprocess all texts
        X_processed = [self.advanced_preprocess(text) for text in X]
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(" Training optimized models with hyperparameter tuning...")
        
        # Create advanced vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.90,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True
        )
        
        # Fit vectorizer and transform data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=chi2, k=5000)
        X_train_selected = self.feature_selector.fit_transform(X_train_vec, y_train)
        X_test_selected = self.feature_selector.transform(X_test_vec)
        
        # Define optimized models with tuned parameters
        models = [
            ('nb', MultinomialNB(alpha=0.1)),
            ('lr', LogisticRegression(C=10.0, random_state=42, max_iter=1000, solver='liblinear')),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)),
            ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
        ]
        
        # Create ensemble with soft voting
        self.model = VotingClassifier(
            estimators=models,
            voting='soft',
            n_jobs=-1
        )
        
        # Train the ensemble model
        print(" Training ensemble model...")
        self.model.fit(X_train_selected, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_selected)
        y_pred_proba = self.model.predict_proba(X_test_selected)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation for robust evaluation
        cv_scores = cross_val_score(self.model, X_train_selected, y_train, cv=5, scoring='f1')
        
        print(" Model trained successfully!")
        print(f" Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f" Test Precision: {precision:.4f} ({precision*100:.1f}%)")
        print(f" Test Recall: {recall:.4f} ({recall*100:.1f}%)")
        print(f" Test F1-Score: {f1:.4f} ({f1*100:.1f}%)")
        print(f" CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        })
        
        self.trained = True
        
        # Save model
        self.save_model()
        
    def classify_email(self, email_text):
        """Classify email with detailed analysis"""
        if not self.trained:
            return {"error": "Model not trained yet"}
        
        try:
            # Preprocess email
            processed_email = self.advanced_preprocess(email_text)
            
            # Extract features for analysis
            features = self.extract_advanced_features(email_text)
            
            # Vectorize and select features
            email_vector = self.vectorizer.transform([processed_email])
            email_selected = self.feature_selector.transform(email_vector)
            
            # Get predictions
            prediction = self.model.predict(email_selected)[0]
            probabilities = self.model.predict_proba(email_selected)[0]
            
            # Calculate detailed scores
            legitimate_prob = probabilities[0] * 100
            spam_prob = probabilities[1] * 100
            confidence = max(probabilities) * 100
            
            # Risk assessment
            risk_level = "LOW"
            if spam_prob > 80:
                risk_level = "HIGH"
            elif spam_prob > 50:
                risk_level = "MEDIUM"
            
            # Feature analysis for explanation
            spam_indicators = []
            if features['money_keywords'] > 0:
                spam_indicators.append(f"Money-related keywords ({features['money_keywords']})")
            if features['urgency_keywords'] > 0:
                spam_indicators.append(f"Urgency keywords ({features['urgency_keywords']})")
            if features['caps_ratio'] > 0.3:
                spam_indicators.append(f"High caps ratio ({features['caps_ratio']:.1%})")
            if features['has_urls'] > 0:
                spam_indicators.append(f"Contains URLs ({features['has_urls']})")
            if features['exclamation_count'] > 2:
                spam_indicators.append(f"Excessive exclamations ({features['exclamation_count']})")
            
            result = {
                "prediction": "Spam" if prediction == 1 else "Legitimate",
                "legitimate_probability": round(legitimate_prob, 1),
                "spam_probability": round(spam_prob, 1),
                "confidence": round(confidence, 1),
                "risk_level": risk_level,
                "spam_indicators": spam_indicators,
                "analysis": {
                    "word_count": features['word_count'],
                    "sentiment": features.get('sentiment_compound', 0),
                    "suspicious_patterns": len(spam_indicators)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Log prediction for analytics
            self.log_prediction(email_text, result)
            
            return result
            
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}
    
    def save_model(self):
        """Save trained model and components"""
        try:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'feature_selector': self.feature_selector,
                'version': self.model_version,
                'training_history': self.training_history
            }
            with open('spam_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            print(" Model saved successfully")
        except Exception as e:
            print(f" Failed to save model: {e}")
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            if os.path.exists('spam_model.pkl'):
                with open('spam_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data['model']
                self.vectorizer = model_data['vectorizer']
                self.feature_selector = model_data['feature_selector']
                self.training_history = model_data.get('training_history', [])
                self.trained = True
                print(" Model loaded successfully")
                return True
        except Exception as e:
            print(f" Failed to load model: {e}")
        return False
    
    def log_prediction(self, email_text, result):
        """Log prediction for analytics"""
        try:
            # Create analytics database if not exists
            conn = sqlite3.connect('analytics.db')
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS predictions
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         timestamp TEXT,
                         email_hash TEXT,
                         prediction TEXT,
                         spam_probability REAL,
                         confidence REAL,
                         risk_level TEXT)''')
            
            # Hash email for privacy
            email_hash = hashlib.sha256(email_text.encode()).hexdigest()[:16]
            
            c.execute('''INSERT INTO predictions 
                        (timestamp, email_hash, prediction, spam_probability, confidence, risk_level)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                     (result['timestamp'], email_hash, result['prediction'], 
                      result['spam_probability'], result['confidence'], result['risk_level']))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Analytics logging failed: {e}")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'spam_classifier_secret_key_2024'

# Global variables
classifier = IndustrySpamClassifier()
training_complete = False

def train_classifier():
    global training_complete
    try:
        # Try to load existing model first
        if not classifier.load_model():
            # Train new model if loading fails
            classifier.train_model()
        training_complete = True
    except Exception as e:
        print(f" Training failed: {e}")

# Start training in background
training_thread = threading.Thread(target=train_classifier)
training_thread.start()

@app.route('/')
def home():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SpamGuard Pro - Industry Email Security</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            --danger-gradient: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
            --warning-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #4a6741 100%);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--primary-gradient);
            min-height: 100vh;
            color: #2d3748;
        }

        .header-section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            padding: 20px 0;
            margin-bottom: 30px;
        }

        .main-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 20px;
        }

        .brand-title {
            color: white;
            font-weight: 800;
            font-size: 2.5rem;
            text-shadow: 0 4px 8px rgba(0,0,0,0.3);
            margin-bottom: 10px;
        }

        .brand-subtitle {
            color: rgba(255, 255, 255, 0.9);
            font-weight: 300;
            font-size: 1.2rem;
        }

        .main-card {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .card-header-custom {
            background: var(--primary-gradient);
            color: white;
            padding: 30px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .card-header-custom::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            opacity: 0.3;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 12px;
            padding: 15px 25px;
            border-radius: 50px;
            font-weight: 600;
            margin: 20px 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .status-training {
            background: rgba(251, 191, 36, 0.2);
            color: #92400e;
        }

        .status-ready {
            background: rgba(16, 185, 129, 0.2);
            color: #065f46;
        }

        .form-control-custom {
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 15px;
            font-size: 16px;
            transition: all 0.3s ease;
            background: #f8fafc;
            resize: vertical;
            min-height: 200px;
        }

        .form-control-custom:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            background: white;
        }

        .btn-analyze {
            background: var(--primary-gradient);
            border: none;
            padding: 15px 30px;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        .btn-analyze:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
            color: white;
        }

        .result-card {
            margin-top: 30px;
            border-radius: 16px;
            overflow: hidden;
            border: none;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .result-header {
            padding: 25px;
            color: white;
            font-weight: 600;
            font-size: 1.3rem;
        }

        .result-spam .result-header {
            background: var(--danger-gradient);
        }

        .result-legitimate .result-header {
            background: var(--success-gradient);
        }

        .probability-section {
            padding: 30px;
            background: #f8fafc;
        }

        .probability-item {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 4px solid;
        }

        .spam-indicator {
            border-left-color: #ef4444;
        }

        .legitimate-indicator {
            border-left-color: #10b981;
        }

        .progress-custom {
            height: 12px;
            border-radius: 6px;
            background: #e2e8f0;
            overflow: hidden;
        }

        .progress-bar-spam {
            background: var(--danger-gradient);
        }

        .progress-bar-legitimate {
            background: var(--success-gradient);
        }

        .analytics-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 25px;
        }

        .analytics-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-top: 3px solid;
        }

        .risk-high { border-top-color: #ef4444; }
        .risk-medium { border-top-color: #f59e0b; }
        .risk-low { border-top-color: #10b981; }

        .example-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 25px;
        }

        .example-card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }

        .example-card:hover {
            border-color: #667eea;
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }

        .example-type {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            padding: 6px 12px;
            border-radius: 20px;
            display: inline-block;
        }

        .example-spam {
            background: rgba(239, 68, 68, 0.1);
            color: #dc2626;
        }

        .example-legitimate {
            background: rgba(16, 185, 129, 0.1);
            color: #059669;
        }

        .spinner-custom {
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }

        .feature-card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-top: 3px solid #667eea;
        }

        .feature-icon {
            font-size: 2.5rem;
            color: #667eea;
            margin-bottom: 15px;
        }

        @media (max-width: 768px) {
            .brand-title { font-size: 2rem; }
            .example-cards { grid-template-columns: 1fr; }
            .main-container { padding: 0 15px; }
        }
    </style>
</head>
<body>
    <div class="header-section">
        <div class="main-container">
            <div class="text-center">
                <h1 class="brand-title">
                    <i class="fas fa-shield-alt"></i> SpamGuard Pro
                </h1>
                <p class="brand-subtitle">Spam Email Classifier</p>
            </div>
        </div>
    </div>

    <div class="main-container">
        <!-- Main Analysis Card -->
        <div class="main-card">
            <div class="card-header-custom">
                <h2><i class="fas fa-search"></i> Email Security Analysis</h2>
                <p>Advanced AI-powered spam detection with real-time analysis</p>
                <div id="status-indicator" class="status-indicator status-training">
                    <div class="spinner-custom"></div>
                    <span>Initializing AI Security Models...</span>
                </div>
            </div>
            
            <div class="card-body p-4">
                <form id="emailForm">
                    <div class="mb-4">
                        <label for="email-content" class="form-label fw-semibold fs-5">
                            <i class="fas fa-envelope text-primary"></i> Email Content Analysis
                        </label>
                        <textarea 
                            id="email-content" 
                            class="form-control form-control-custom" 
                            placeholder="Paste your email content here for comprehensive security analysis..."
                            disabled
                            rows="8"
                        ></textarea>
                        <div class="form-text">
                            <i class="fas fa-info-circle"></i> 
                            Your email content is processed securely and never stored permanently.
                        </div>
                    </div>
                    
                    <div class="d-grid gap-2 d-md-flex justify-content-md-start">
                        <button type="button" id="analyze-btn" class="btn btn-analyze" disabled onclick="analyzeEmail()">
                            <i class="fas fa-shield-alt"></i>
                            <span>Analyze Security Risk</span>
                        </button>
                        <button type="button" class="btn btn-outline-secondary" onclick="clearForm()">
                            <i class="fas fa-eraser"></i> Clear
                        </button>
                    </div>
                </form>

                <div id="result-section" style="display: none;"></div>
            </div>
        </div>

        <!-- Features Section -->
        <div class="main-card">
            <div class="card-body p-4">
                <h3 class="text-center mb-4">
                    <i class="fas fa-cogs text-primary"></i> Advanced Security Features
                </h3>
                
                <div class="feature-grid">
                    <div class="feature-card">
                        <div class="feature-icon">
                            <i class="fas fa-brain"></i>
                        </div>
                        <h5>AI Ensemble Learning</h5>
                        <p class="text-muted">5 advanced algorithms working together for maximum accuracy</p>
                    </div>
                    
                    <div class="feature-card">
                        <div class="feature-icon">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <h5>Real-time Analysis</h5>
                        <p class="text-muted">Instant threat assessment with detailed probability scoring</p>
                    </div>
                    
                    <div class="feature-card">
                        <div class="feature-icon">
                            <i class="fas fa-microscope"></i>
                        </div>
                        <h5>Deep Pattern Recognition</h5>
                        <p class="text-muted">Advanced feature extraction and linguistic analysis</p>
                    </div>
                    
                    <div class="feature-card">
                        <div class="feature-icon">
                            <i class="fas fa-lock"></i>
                        </div>
                        <h5>Privacy Protected</h5>
                        <p class="text-muted">Secure processing with no permanent data storage</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Examples Section -->
        <div class="main-card">
            <div class="card-body p-4">
                <h3 class="text-center mb-4">
                    <i class="fas fa-flask text-primary"></i> Test Examples
                </h3>
                
                <div class="example-cards">
                    <div class="example-card" onclick="fillExample(this)">
                        <div class="example-type example-spam">High Risk Spam</div>
                        <p><strong>Subject:</strong> URGENT: Claim Your $50,000 Prize NOW!</p>
                        <p class="text-muted mb-0">CONGRATULATIONS! You have been selected as a WINNER in our international lottery! Click here immediately to claim your $50,000 prize before it expires! Act now - limited time offer!</p>
                    </div>
                    
                    <div class="example-card" onclick="fillExample(this)">
                        <div class="example-type example-legitimate">Legitimate Business</div>
                        <p><strong>Subject:</strong> Quarterly Meeting Reminder</p>
                        <p class="text-muted mb-0">Hi John, this is a reminder about our quarterly team meeting scheduled for tomorrow at 2 PM in conference room A. Please bring your project reports. Thanks!</p>
                    </div>
                    
                    <div class="example-card" onclick="fillExample(this)">
                        <div class="example-type example-spam">Phishing Attempt</div>
                        <p><strong>Subject:</strong> Account Security Alert</p>
                        <p class="text-muted mb-0">URGENT: Your account will be suspended! Verify your login credentials immediately by clicking this link. Failure to act within 24 hours will result in permanent account closure.</p>
                    </div>
                    
                    <div class="example-card" onclick="fillExample(this)">
                        <div class="example-type example-legitimate">Personal Communication</div>
                        <p><strong>Subject:</strong> Weekend Plans</p>
                        <p class="text-muted mb-0">Hey! Are you free this weekend? I was thinking we could check out that new restaurant downtown. Let me know what you think!</p>
                    </div>
                </div>
            </div>
        </div>

      
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <script>
        let trainingComplete = false;
        let analysisHistory = [];

        // Check training status
        function checkTrainingStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    if (data.training_complete && !trainingComplete) {
                        trainingComplete = true;
                        updateUI();
                    }
                })
                .catch(error => console.error('Error:', error));
        }

        function updateUI() {
            const statusDiv = document.getElementById('status-indicator');
            const emailInput = document.getElementById('email-content');
            const analyzeBtn = document.getElementById('analyze-btn');

            statusDiv.className = 'status-indicator status-ready';
            statusDiv.innerHTML = '<i class="fas fa-check-circle"></i> <span>AI Security System Online - Ready for Analysis</span>';
            
            emailInput.disabled = false;
            analyzeBtn.disabled = false;
        }

        function analyzeEmail() {
            const emailContent = document.getElementById('email-content').value.trim();
            if (!emailContent) {
                showAlert('Please enter email content to analyze.', 'warning');
                return;
            }

            if (emailContent.length < 10) {
                showAlert('Please enter a more substantial email content for accurate analysis.', 'warning');
                return;
            }

            const analyzeBtn = document.getElementById('analyze-btn');
            const originalContent = analyzeBtn.innerHTML;
            analyzeBtn.innerHTML = '<div class="spinner-custom"></div> <span>Analyzing...</span>';
            analyzeBtn.disabled = true;

            fetch('/api/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email: emailContent })
            })
            .then(response => response.json())
            .then(data => {
                displayResult(data);
                analysisHistory.push(data);
                analyzeBtn.innerHTML = originalContent;
                analyzeBtn.disabled = false;
            })
            .catch(error => {
                console.error('Error:', error);
                analyzeBtn.innerHTML = originalContent;
                analyzeBtn.disabled = false;
                showAlert('Analysis failed. Please try again.', 'danger');
            });
        }

        function displayResult(data) {
            if (data.error) {
                showAlert('Analysis Error: ' + data.error, 'danger');
                return;
            }

            const resultSection = document.getElementById('result-section');
            const isSpam = data.prediction === 'Spam';
            
            let indicatorsHtml = '';
            if (data.spam_indicators && data.spam_indicators.length > 0) {
                indicatorsHtml = `
                    <div class="mt-3">
                        <h6><i class="fas fa-exclamation-triangle text-warning"></i> Detected Risk Indicators:</h6>
                        <ul class="list-unstyled">
                            ${data.spam_indicators.map(indicator => `<li><i class="fas fa-chevron-right text-muted"></i> ${indicator}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            resultSection.innerHTML = `
                <div class="result-card ${isSpam ? 'result-spam' : 'result-legitimate'}">
                    <div class="result-header">
                        <i class="fas ${isSpam ? 'fa-exclamation-triangle' : 'fa-shield-check'}"></i>
                        Security Assessment: ${data.prediction}
                        <div class="float-end">
                            <span class="badge ${isSpam ? 'bg-danger' : 'bg-success'}">
                                Risk: ${data.risk_level}
                            </span>
                        </div>
                    </div>
                    
                    <div class="probability-section">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="probability-item spam-indicator">
                                    <div class="d-flex justify-content-between align-items-center mb-2">
                                        <span class="fw-semibold">
                                            <i class="fas fa-exclamation-triangle text-danger"></i>
                                            Spam Probability
                                        </span>
                                        <span class="fw-bold text-danger">${data.spam_probability}%</span>
                                    </div>
                                    <div class="progress progress-custom">
                                        <div class="progress-bar progress-bar-spam" 
                                             style="width: ${data.spam_probability}%" 
                                             aria-valuenow="${data.spam_probability}" 
                                             aria-valuemin="0" 
                                             aria-valuemax="100">
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="probability-item legitimate-indicator">
                                    <div class="d-flex justify-content-between align-items-center mb-2">
                                        <span class="fw-semibold">
                                            <i class="fas fa-shield-check text-success"></i>
                                            Legitimate Probability
                                        </span>
                                        <span class="fw-bold text-success">${data.legitimate_probability}%</span>
                                    </div>
                                    <div class="progress progress-custom">
                                        <div class="progress-bar progress-bar-legitimate" 
                                             style="width: ${data.legitimate_probability}%" 
                                             aria-valuenow="${data.legitimate_probability}" 
                                             aria-valuemin="0" 
                                             aria-valuemax="100">
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="analytics-section">
                            <div class="analytics-card risk-${data.risk_level.toLowerCase()}">
                                <div class="fw-bold fs-4">${data.confidence}%</div>
                                <div class="text-muted">Analysis Confidence</div>
                            </div>
                            
                            <div class="analytics-card">
                                <div class="fw-bold fs-4">${data.analysis.word_count}</div>
                                <div class="text-muted">Words Analyzed</div>
                            </div>
                            
                            <div class="analytics-card">
                                <div class="fw-bold fs-4">${data.analysis.suspicious_patterns}</div>
                                <div class="text-muted">Risk Patterns</div>
                            </div>
                            
                            <div class="analytics-card">
                                <div class="fw-bold fs-4">${Math.abs(data.analysis.sentiment).toFixed(2)}</div>
                                <div class="text-muted">Sentiment Score</div>
                            </div>
                        </div>
                        
                        ${indicatorsHtml}
                        
                        <div class="text-center mt-4">
                            <small class="text-muted">
                                <i class="fas fa-clock"></i> 
                                Analysis completed at ${new Date(data.timestamp).toLocaleString()}
                            </small>
                        </div>
                    </div>
                </div>
            `;
            
            resultSection.style.display = 'block';
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        function fillExample(element) {
            const textElement = element.querySelector('.text-muted');
            const text = textElement.textContent;
            document.getElementById('email-content').value = text;
            
            // Scroll to form
            document.getElementById('email-content').scrollIntoView({ behavior: 'smooth' });
            document.getElementById('email-content').focus();
        }

        function clearForm() {
            document.getElementById('email-content').value = '';
            document.getElementById('result-section').style.display = 'none';
        }

        function showAlert(message, type) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
            alertDiv.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            
            const container = document.querySelector('.main-container');
            container.insertBefore(alertDiv, container.firstChild);
            
            setTimeout(() => {
                alertDiv.remove();
            }, 5000);
        }

        // Start checking training status
        const statusInterval = setInterval(() => {
            if (!trainingComplete) {
                checkTrainingStatus();
            } else {
                clearInterval(statusInterval);
            }
        }, 2000);

        // Initial check
        checkTrainingStatus();

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                if (trainingComplete && !document.getElementById('analyze-btn').disabled) {
                    analyzeEmail();
                }
            }
        });
    </script>
</body>
</html>
    ''')

@app.route('/api/status')
def status():
    return jsonify({
        'training_complete': training_complete,
        'model_version': classifier.model_version,
        'features_count': len(classifier.spam_keywords) if hasattr(classifier, 'spam_keywords') else 0
    })

@app.route('/api/classify', methods=['POST'])
def classify():
    data = request.json
    email_content = data.get('email', '')
    
    if not training_complete:
        return jsonify({'error': 'AI models are still training. Please wait a moment.'})
    
    if not email_content.strip():
        return jsonify({'error': 'Please provide email content to analyze.'})
    
    result = classifier.classify_email(email_content)
    return jsonify(result)

@app.route('/api/analytics')
def analytics():
    """Get prediction analytics"""
    try:
        conn = sqlite3.connect('analytics.db')
        c = conn.cursor()
        
        # Get recent predictions
        c.execute('''SELECT prediction, COUNT(*) as count 
                    FROM predictions 
                    WHERE timestamp > datetime('now', '-7 days')
                    GROUP BY prediction''')
        
        results = c.fetchall()
        conn.close()
        
        analytics_data = {
            'recent_predictions': dict(results),
            'total_predictions': sum([count for _, count in results])
        }
        
        return jsonify(analytics_data)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting SpamGuard Pro - Industry Email Security System...")
    print("🤖 Initializing Advanced AI Models...")
    print("🌐 Web Interface: http://127.0.0.1:5000")
    print("🔒 Security Features: Ensemble Learning, Real-time Analysis, Privacy Protection")
    app.run(debug=True, host='0.0.0.0', port=5000)
