# Email Spam Classification Using Logistic Regression

## Project Overview
This project implements an email spam classification system using Logistic Regression. The application features a user-friendly web interface built with Streamlit, allowing users to easily classify emails as spam or legitimate.

## Features
- Real-time email classification
- Interactive web interface
- Visual confidence indicators
- Pre-trained model using UCI ML Repository's SMS Spam Collection
- High accuracy classification

## Technologies Used
- Python 3.8+
- Scikit-learn for machine learning
- Streamlit for web interface
- Pandas for data processing
- TF-IDF Vectorization for text processing

## Technical Implementation
1. **Data Processing**
   - Uses TF-IDF Vectorization for text feature extraction
   - Implements binary classification (spam/not spam)
   - Includes data splitting and preprocessing

2. **Model**
   - Logistic Regression classifier
   - Features automatic model training
   - Provides probability scores for predictions

3. **User Interface**
   - Clean, intuitive web interface
   - Real-time classification
   - Visual confidence indicators
   - Detailed analysis results

## Setup Instructions

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run spam_classifier.py
```

3. Access the web interface at `http://localhost:8501`

## Usage Guide
1. Launch the application
2. Wait for the model to train (happens automatically on first launch)
3. Enter or paste email text into the text area
4. Click "Classify Email" to get results
5. View classification results and confidence scores

## Performance
- The model achieves high accuracy on the test dataset
- Real-time classification performance
- Robust handling of various email formats

## Author Group Information
[Add your group member information here]
- Student 1 (ID: 2101140062)
- Student 2 (ID: xxx)
- Student 3 (ID: xxx)
