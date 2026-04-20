from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import re

app = Flask(__name__)
app.secret_key = "fakenewssecret"

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

USERNAME = "admin"
PASSWORD = "1234"

# -----------------------
# Text cleaning (same as training)
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username")
        pwd = request.form.get("password")

        if user == USERNAME and pwd == PASSWORD:
            session["user"] = user
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid login")

    return render_template("login.html")


@app.route("/home", methods=["GET", "POST"])
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None

    if request.method == "POST":
        news = request.form.get("news", "").strip()

        if len(news) < 10:
            prediction = "❌ Please enter meaningful news text"
        else:
            cleaned = clean_text(news)
            vec = vectorizer.transform([cleaned])

            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]
            confidence = max(prob) * 100

            if pred == 1:
                prediction = f"🟢 Real News ({confidence:.2f}% confidence)"
            else:
                prediction = f"🔴 Fake News ({confidence:.2f}% confidence)"

    return render_template("index.html", prediction=prediction)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
