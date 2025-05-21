from flask import Flask, request, render_template, jsonify
from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering
from transformers import pipeline
import torch
import cv2
import numpy as np
import pytesseract
import re
import random
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Initialize the Flask app
app = Flask(__name__)

# Load pre-trained models from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
ner_tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
ner_model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Set up pipelines for NER and QA
ner_pipe = pipeline('ner', model=ner_model, tokenizer=ner_tokenizer)
qa_pipe = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

# Firebase initialization
cred = credentials.Certificate("firebase_config.json")  # Add your Firebase service account key
firebase_admin.initialize_app(cred)
db = firestore.client()

# Define the legal clauses patterns
CLAUSES = {
    "termination": r"(termination.*?)(?:\n|\.)",
    "indemnity": r"(indemnity.*?)(?:\n|\.)",
    "governing law": r"(governing law.*?)(?:\n|\.)"
}

# Put the classification model in evaluation mode
model.eval()

def classify_text_with_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class
    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

def extract_entities(text):
    entities = ner_pipe(text)
    named_entities = []
    for entity in entities:
        named_entities.append({
            'entity': entity['word'],
            'label': entity['entity'],
            'score': entity['score']
        })
    return named_entities

def answer_question(context, question):
    result = qa_pipe(question=question, context=context)
    return result['answer']

def generate_quiz(clauses):
    questions = []
    for key, clause in clauses.items():
        options = [clause, "Defines party liabilities", "Specifies arbitration rules", "Outlines payment terms"]
        random.shuffle(options)
        questions.append({
            "question": f"What does the '{key.title()}' clause refer to?",
            "options": options,
            "answer": clause
        })
    return questions

def save_to_firebase(text, quiz, user_id="anonymous"):
    db.collection("scans").add({
        "user_id": user_id,
        "text": text,
        "quiz": quiz,
        "xp": 10 * len(quiz),
        "timestamp": datetime.utcnow().isoformat()
    })

    user_ref = db.collection("users").document(user_id)
    user_doc = user_ref.get()
    current_xp = user_doc.to_dict()["xp"] if user_doc.exists else 0
    user_ref.set({"xp": current_xp + 10 * len(quiz)}, merge=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/scan", methods=["POST"])
def scan_document():
    file = request.files["document"]
    user_id = request.form.get("user_id", "anonymous")
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    text = pytesseract.image_to_string(image)

    # Classify the extracted text using BERT
    classification_result = classify_text_with_bert(text)

    # Extract clauses using regex
    extracted_clauses = {}
    for key, pattern in CLAUSES.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_clauses[key] = match.group(1)

    # Generate quiz based on extracted clauses
    quiz = generate_quiz(extracted_clauses)

    # Extract entities using NER
    entities = extract_entities(text)

    # Answer a sample question using QA model
    sample_question = "What is the termination clause about?"
    answer = answer_question(text, sample_question)

    # Save the scan and quiz data to Firebase
    save_to_firebase(text, quiz, user_id)

    return render_template("results.html", clauses=extracted_clauses, quiz=quiz, classification=classification_result, entities=entities, answer=answer)

@app.route("/dashboard")
def dashboard():
    user_id = request.args.get("user", "anonymous")
    user_ref = db.collection("users").document(user_id)
    user_data = user_ref.get().to_dict()

    xp = user_data["xp"] if user_data else 0
    level = xp // 100
    progress = xp % 100

    return render_template("dashboard.html", xp=xp, level=level, progress=progress)

if __name__ == "__main__":
    app.run(debug=True)
