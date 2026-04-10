# Emotion Detection Pipeline

An advanced end-to-end Machine Learning web application designed to analyze text for emotional context and extract deep behavioral patterns over time. 

Powered by a fine-tuned Hugging Face `DistilRoBERTa` model, this project moves beyond simple sentiment analysis by tracking a continuous emotional timeline, mapping human nature through Markov chain transition tracking, volatility detection, and contextual data analysis.

## 🚀 Features

### 1. Real-Time Emotion Classification
*   Analyses unstructured text (chat messages, diary entries, etc.) using `j-hartmann/emotion-english-distilroberta-base`.
*   Classifies text into 7 core emotions: **anger**, **disgust**, **fear**, **joy**, **neutral**, **sadness**, and **surprise**.
*   Real-time processing with live confidence scores.

### 2. Advanced Behavior Pattern Engine (The USP)
Instead of just classifying text, the system learns and tracks the user's emotional patterns via an intelligent rule/ML-hybrid behavior engine:
*   **Markov Chain Transitions**: Tracks how emotions flow (e.g., probability of `sadness` shifting to `anger`).
*   **Trend & Volatility**: Calculates moving averages of "emotional valence" to detect stability, improvement, or decline.
*   **Context Windows**: Summarizes short-term emotional states across the newest entries.
*   **Micro-Spike Detection**: Flags rapid outbursts of single emotions.
*   **Human Nature Insight Generator**: Automatically generates readable alerts (e.g., *"Your emotions have been highly unstable"* or *"Possible stress pattern"*).

### 3. Dataset Integration & Exploration
Natively integrated with world-class emotion datasets for future training and current bench-marking:
*   **EmotionLines / MELD**: Automatically loads multi-party conversational emotion datasets directly from CSV.
*   **GoEmotions**: Maps HuggingFace’s 28-class Reddit dataset down into the 7-class schema for compatibility and bulk analysis.
*   **Dataset Explorer UI**: A responsive, paginated interface to browse dataset records, analyze model accuracy versus ground truth, and check label distribution.

### 4. Interactive UI
*   Polished **React + Vite** frontend.
*   Glassmorphism design aesthetic with smooth micro-animations.
*   Comprehensive `Chart.js` integration (Timeline trends, Transition bar charts, Pie distributions).

---

## 🛠️ Technology Stack

**Backend**
*   **Python 3.14+**
*   **FastAPI**: High-performance async REST API.
*   **HuggingFace Transformers**: Local inference via PyTorch (`distilroberta-base`).
*   **SQLAlchemy (SQLite)**: Durable timeline storage.
*   **Pandas & Scikit-Learn**: For dataset processing and behavioral data manipulation.
*   **Uvicorn**: ASGI Server.

**Frontend**
*   **React + Vite**: Fast UI rendering.
*   **Vanilla CSS**: Custom styling with CSS Variables.
*   **Chart.js / react-chartjs-2**: High-quality data visualization.

---

## 💻 Getting Started (Local Development)

### 1. Backend Setup
Navigate to the `backend/` directory, set up your Python environment, and start the Fast API server.

```bash
cd backend
python -m venv venv

# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*The API will be available at `http://localhost:8000`. Swagger API docs can be found at `http://localhost:8000/docs`.*

### 2. Frontend Setup
In a new terminal window, navigate to the `frontend/` directory, install Node dependencies, and start the Vite dev server.

```bash
cd frontend
npm install
npm run dev
```
*The web application will be available at `http://localhost:5173/`.*

---

## 📈 Analyzing the Datasets

To test the model's performance on established public datasets:
1. Open the application.
2. Navigate to the **Datasets** tab.
3. Click **Load MELD** or **Load GoEmotions**.
4. The background thread will download the dataset, process it through the active BERT model, and populate the explorer UI with metrics.

---
*Created as an advanced exploration of Applied Artificial Intelligence and human-behavior tracking in conversational agents.*
