# 🧠 Health Intelligence System API

A Flask-based AI-powered health intelligence system that analyzes lab reports and wearable data to predict disease risks.

---

## 🚀 Features

* 🧬 Lab report OCR (PDF & images)
* 📊 AI-based risk prediction (ML models)
* ❤️ Disease prediction engine (10+ diseases)
* ⌚ Fitbit integration (OAuth2)
* 📈 Patient history tracking
* 🌐 REST API for mobile/web apps

---

## 🛠️ Tech Stack

* Python
* Flask
* Scikit-learn / XGBoost
* SHAP (Explainable AI)
* Tesseract OCR
* Fitbit API

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
python api_server.py
```

---

## 🔗 API Endpoints

| Method | Endpoint           | Description        |
| ------ | ------------------ | ------------------ |
| POST   | /api/register      | Register user      |
| POST   | /api/login         | Login              |
| POST   | /api/analyze/{id}  | Run analysis       |
| GET    | /api/report/{id}   | Get report         |
| GET    | /api/diseases/{id} | Disease prediction |

---

## 📌 Notes

* Run `med.py` first to train the model
* Configure Fitbit credentials for wearable sync

---

## 👨‍💻 Authors

* Mohammed Ameer
* Team PSNA AI & DS

---
