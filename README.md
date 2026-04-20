📰 Fake News Detection System

📌 Overview
This project is a Machine Learning-based web application that detects whether a news article is **Real or Fake** using Natural Language Processing (NLP) techniques. It helps reduce misinformation by providing quick and reliable predictions.

 🚀 Features
- Real-time fake news detection  
- Text preprocessing (cleaning, stopword removal, tokenization)  
- TF-IDF feature extraction  
- Logistic Regression classification  
- Confidence score display  
- Simple Flask-based web interface  

🛠️ Tech Stack
Python | Scikit-learn | NLTK | Flask | TF-IDF | Joblib  

📂 Project Structure
data/ – dataset files  
model/ – trained model (.pkl)  
templates/ – HTML pages  
static/ – CSS/JS  
app.py – Flask app  
train.py – model training  

⚙️ Installation
1. Clone repo: `git clone <your-repo-link>`  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run app: `python app.py`  
4. Open: `http://127.0.0.1:5000/`  

🧠 Working
User inputs news → preprocessing → TF-IDF → model prediction → result (Real/Fake with confidence).

📊 Performance
Evaluated using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

⚠️ Limitations
- Works only for text-based news  
- Accuracy depends on dataset quality  

🔮 Future Scope
- Add Deep Learning (LSTM, BERT)  
- Multi-language support  
- Deploy on cloud  

Author
Harinakshi | MCA Student  

Note
Feel free to fork, contribute, and improve this project!
