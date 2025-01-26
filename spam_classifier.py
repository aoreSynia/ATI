import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle
import os
import io
import base64
import requests
import zipfile
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Page config
st.set_page_config(page_title="Enhanced Email Spam Classifier", page_icon="📧", layout="wide")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'history' not in st.session_state:
    st.session_state.history = []

def preprocess_text(text):
    """Enhanced text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def load_data():
    """Load and preprocess the spam dataset"""
    try:
        # Download dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        response = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(response.content))
        
        # Read the dataset
        with z.open('SMSSpamCollection') as f:
            content = f.read().decode('utf-8')
            lines = content.strip().split('\n')
            data = [line.strip().split('\t') for line in lines]
            df = pd.DataFrame(data, columns=['label', 'message'])
            
            # Preprocess messages
            df['processed_message'] = df['message'].apply(preprocess_text)
            return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def optimize_model(X_train, y_train):
    """Optimize logistic regression model using cross-validation"""
    best_score = 0
    best_params = {}
    
    # Test different parameters
    for C in [0.1, 1.0, 10.0]:
        for solver in ['liblinear', 'saga']:
            model = LogisticRegression(C=C, solver=solver, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=5)
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = {'C': C, 'solver': solver}
    
    # Create model with best parameters
    return LogisticRegression(**best_params, random_state=42)

def train_model(df):
    """Train and evaluate the model"""
    # Prepare data
    X = df['processed_message']
    y = (df['label'] == 'spam').astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Optimize and train model
    model = optimize_model(X_train_vectorized, y_train)
    model.fit(X_train_vectorized, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_vectorized)
    performance_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return model, vectorizer, performance_metrics

def get_important_features(text, prediction, model, vectorizer):
    """Get important features that influenced the prediction"""
    # Get feature names and coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Get TF-IDF scores for the input text
    text_vectorized = vectorizer.transform([text])
    feature_scores = text_vectorized.toarray()[0]
    
    # Calculate importance scores
    importance = coefficients * feature_scores
    
    # Get top features
    top_indices = np.argsort(np.abs(importance))[-10:]
    top_features = [(feature_names[i], importance[i]) for i in top_indices if importance[i] != 0]
    
    return sorted(top_features, key=lambda x: abs(x[1]), reverse=True)

def create_word_cloud(text_series, title):
    """Create and return a word cloud image"""
    text = ' '.join(text_series)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    
    return fig

def plot_confusion_matrix(conf_matrix):
    """Create confusion matrix plot"""
    labels = ['Not Spam', 'Spam']
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=labels,
        y=labels,
        text=conf_matrix,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label'
    )
    
    return fig

def process_uploaded_file(uploaded_file):
    """Process uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        if 'message' not in df.columns:
            st.error("CSV file must contain a 'message' column")
            return None
        return df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def get_csv_download_link(df, filename):
    """Create a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def main():
    st.title("Enhanced Email Spam Classification")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("""
    This enhanced spam classification system uses machine learning to identify spam emails.
    Features include:
    - Single email classification
    - Batch processing via CSV
    - Detailed analysis and visualization
    - Performance metrics
    """)
    
    # Initialize model
    if st.session_state.model is None:
        with st.spinner("Loading and training model..."):
            df = load_data()
            if df is not None:
                model, vectorizer, metrics = train_model(df)
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                st.session_state.metrics = metrics
                st.success(f"Model trained successfully! Accuracy: {metrics['accuracy']:.2%}")
    
    # Main content area
    tabs = st.tabs(["Single Email", "Batch Processing", "Model Performance", "Classification History"])
    
    # Single Email Classification
    with tabs[0]:
        st.header("Single Email Classification")
        email_text = st.text_area("Enter email text:", height=150)
        
        if st.button("Classify Email"):
            if email_text.strip() == "":
                st.warning("Please enter some text to classify.")
            else:
                # Preprocess and classify
                processed_text = preprocess_text(email_text)
                text_vectorized = st.session_state.vectorizer.transform([processed_text])
                prediction = st.session_state.model.predict(text_vectorized)[0]
                probability = st.session_state.model.predict_proba(text_vectorized)[0]
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Classification", "SPAM" if prediction == 1 else "NOT SPAM")
                with col2:
                    st.metric("Spam Probability", f"{probability[1]:.2%}")
                
                # Show importance features
                st.subheader("Important Features")
                important_features = get_important_features(
                    processed_text,
                    prediction,
                    st.session_state.model,
                    st.session_state.vectorizer
                )
                
                feature_df = pd.DataFrame(
                    important_features,
                    columns=['Feature', 'Importance']
                )
                st.table(feature_df)
                
                # Add to history
                st.session_state.history.append({
                    'timestamp': datetime.now(),
                    'content': email_text,
                    'prediction': "SPAM" if prediction == 1 else "NOT SPAM",
                    'probability': probability[1]
                })
    
    # Batch Processing
    with tabs[1]:
        st.header("Batch Processing")
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file is not None:
            df = process_uploaded_file(uploaded_file)
            if df is not None:
                # Process all emails
                with st.spinner("Processing emails..."):
                    df['processed_text'] = df['message'].apply(preprocess_text)
                    vectors = st.session_state.vectorizer.transform(df['processed_text'])
                    df['prediction'] = st.session_state.model.predict(vectors)
                    df['spam_probability'] = st.session_state.model.predict_proba(vectors)[:, 1]
                    
                    # Display results
                    results_df = df[['message', 'prediction', 'spam_probability']].copy()
                    results_df['prediction'] = results_df['prediction'].map({1: 'SPAM', 0: 'NOT SPAM'})
                    st.dataframe(results_df)
                    
                    # Create download link
                    st.markdown(get_csv_download_link(results_df, "classification_results.csv"), unsafe_allow_html=True)
                    
                    # Show distribution plot
                    fig = px.pie(
                        results_df,
                        names='prediction',
                        title='Distribution of Classifications'
                    )
                    st.plotly_chart(fig)
    
    # Model Performance
    with tabs[2]:
        st.header("Model Performance")
        
        # Display metrics
        st.subheader("Classification Report")
        st.text(st.session_state.metrics['report'])
        
        # Show confusion matrix
        st.subheader("Confusion Matrix")
        conf_matrix_plot = plot_confusion_matrix(st.session_state.metrics['confusion_matrix'])
        st.plotly_chart(conf_matrix_plot)
    
    # Classification History
    with tabs[3]:
        st.header("Classification History")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df)
            
            # Download history
            st.markdown(get_csv_download_link(history_df, "classification_history.csv"), unsafe_allow_html=True)
        else:
            st.info("No classification history yet. Classify some emails to see them here.")

if __name__ == "__main__":
    main()
