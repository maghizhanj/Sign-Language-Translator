print("Wait for 1 Minute For the Server to Load and To Configure the Camera")
from flask import Flask, render_template, Response, jsonify, request
import cv2
import csv
import joblib
import pandas as pd
import pyttsx3
from collections import Counter
from cvzone.HandTrackingModule import HandDetector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
app = Flask(__name__)
try:
    model = joblib.load("sign_model.pkl")
except:
    model = None
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1,detectionCon=0.8)
current_prediction = ""
sentence = ""
predictions = []
STABLE_FRAMES = 10
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 100)
    engine.say(text)
    engine.runAndWait()
def train_sign(label):
    global model
    samples = 0
    with open("sign_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        while samples < 300:
            success, img = cap.read()
            if not success:
                continue
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                landmarks = hand["lmList"]
                x0 = landmarks[0][0]
                y0 = landmarks[0][1]
                row = []
                for lm in landmarks:
                    row.append(lm[0] - x0)
                    row.append(lm[1] - y0)
                row.append(label)
                writer.writerow(row)
                samples += 1
                cv2.putText(img,f"{label}: {samples}/300",(20, 60),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)
                cv2.imshow("Training", img)
                cv2.waitKey(1)
    cv2.destroyAllWindows()
    data = pd.read_csv("sign_data.csv",header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=300,random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,pred)
    joblib.dump(model,"sign_model.pkl")
    return accuracy
def generate_frames():
    global current_prediction
    global predictions
    while True:
        success, img = cap.read()
        if not success:
            continue
        hands, img = detector.findHands(img)
        if hands and model is not None:
            hand = hands[0]
            landmarks = hand["lmList"]
            x0 = landmarks[0][0]
            y0 = landmarks[0][1]
            row = []
            for lm in landmarks:
                row.append(lm[0] - x0)
                row.append(lm[1] - y0)
            prediction = model.predict([row])[0]
            predictions.append(prediction)
            if len(predictions) > STABLE_FRAMES:
                predictions.pop(0)
            current_prediction = Counter(
                predictions
            ).most_common(1)[0][0]
            confidence = 0
            try:
                confidence = max(
                    model.predict_proba([row])[0]
                )
            except:
                pass
            cv2.putText(img,f"{current_prediction} ({confidence:.2f})",(20, 60),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+ frame +b'\r\n')
@app.route("/")
def dashboard():
    return render_template("dashboard.html")
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )
@app.route("/get-current-sign")
def get_current_sign():
    return jsonify({"sign": current_prediction})
@app.route("/add-character")
def add_character():
    global sentence
    if current_prediction:
        sentence += current_prediction
    return jsonify({"sentence": sentence})
@app.route("/add-space")
def add_space():
    global sentence
    sentence += " "
    return jsonify({"sentence": sentence})
@app.route("/backspace")
def backspace():
    global sentence
    sentence = sentence[:-1]
    return jsonify({"sentence": sentence})
@app.route("/clear-sentence")
def clear_sentence():
    global sentence
    sentence = ""
    return jsonify(
        {"sentence": sentence})
@app.route("/get-sentence")
def get_sentence():
    return jsonify(
        {"sentence": sentence})
@app.route("/speak")
def speak():
    if sentence:
        speak_text(sentence)
    return jsonify(
        {"sentence": sentence})
@app.route(
    "/text-to-sign",
    methods=["GET", "POST"]
)
def text_to_sign():
    sign_images = []
    submitted_text = ""
    if request.method == "POST":
        submitted_text = request.form.get("user_text","")
        for char in submitted_text.lower():
            if char.isalpha():
                sign_images.append(f"signs/{char}.png")
            elif char == " ":
                sign_images.append("space")
    return render_template("text_to_sign.html",sign_images=sign_images,submitted_text=submitted_text)
@app.route("/train-sign")
def train_page():
    return render_template("train_sign.html")

@app.route(
    "/train-model",methods=["POST"])
def train_model():
    label = request.form.get("label")
    if not label:
        return jsonify({"error":"No label supplied"})
    accuracy = train_sign(label.upper())
    return jsonify(
        {"status": "success",
        "accuracy":accuracy*100})
@app.route("/get-labels")
def get_labels():
    try:
        data = pd.read_csv("sign_data.csv",header=None)
        labels = sorted(data.iloc[:, -1].unique().tolist())
        return jsonify(labels)
    except:
        return jsonify([])
@app.route("/delete-label",methods=["POST"])
def delete_label():
    global model
    label = request.form.get("label","").upper()
    try:
        data = pd.read_csv("sign_data.csv",header=None)
        data = data[data.iloc[:, -1] != label]
        data.to_csv("sign_data.csv",index=False,header=False)
        if len(data) > 0:
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            model = RandomForestClassifier(n_estimators=300,random_state=42)
            model.fit(X,y)
            joblib.dump(model,"sign_model.pkl")
        return jsonify({"message":f"{label} deleted successfully"})
    except Exception as e:
        return jsonify({"message":str(e)})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)