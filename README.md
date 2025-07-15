# 🗑️ Smart Industrial Waste Monitoring with Active Learning

## 🔍 Overview

This project is a smart, self-learning waste management system for industrial garbage bins using **IR sensors** and **Machine Learning**. It predicts whether a bin is full based on the waste level and **continuously improves itself** through an **automated active learning loop** — no human input needed.

---

## 🧠 Problem Statement

Industrial garbage bins can overflow if not emptied on time, leading to inefficiency and health hazards. Manual checks are not scalable. The goal is to automate this process using real-time data and machine learning — and make the system smarter over time.

---

## ⚙️ How It Works

### 🧾 Sensor Setup

- **2 IR Sensors**:
  - Top sensor: Measures distance from top of bin to waste surface.
  - Bottom sensor: Verifies empty state.
- Sensors log data periodically in CSV format.

### 🤖 ML Pipeline

1. **Classification Model**:
   - Input: `Waste Level (cm)`
   - Output: `is_full` (1 = full, 0 = not full)
   - Model: `RandomForestClassifier` from `scikit-learn`

2. **Active Learning**:
   - Uses model confidence (`predict_proba`) to identify high-certainty predictions.
   - Automatically adds confident predictions to training data.
   - Retrains the model with this augmented dataset.

---

## 🔁 Active Learning Logic

- Confident predictions (≥ 95%) are considered trustworthy.
- These samples are appended to the training set automatically.
- The model is retrained periodically to adapt to real-world conditions.
- **No human labeling or review required.**

---

## 🛠️ Tech Stack

| Component       | Technology        |
|----------------|-------------------|
| Sensors         | IR Proximity Sensors |
| Controller      | Arduino / ESP32      |
| ML Model        | Random Forest         |
| Language        | Python 3.x            |
| Libraries       | `pandas`, `scikit-learn`, `joblib`, `numpy` |
| Data Format     | CSV                   |

---

## 📂 Project Structure

waste-bin-predictor/
├── waste_data.csv # Initial training data
├── waste_testing_data.csv # New incoming sensor readings
├── waste_predictions.csv # Output with predictions and confidence
├── waste_management_model.pkl # Trained ML model
├── main.py # Active learning + prediction script
├── README.md # This file
