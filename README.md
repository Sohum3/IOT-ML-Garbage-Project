# 🗑️ Smart Industrial Waste Bin Monitoring System

## 🔍 Overview
This project combines **IoT** and **Machine Learning** to monitor and predict the optimal time to empty a large **industrial garbage bin** using two **IR sensors** and a **Random Forest Classifier**.

It eliminates overflow issues and improves efficiency in industrial waste collection by classifying whether a bin is **full or not**, based on real-time sensor data.

---

## 🧠 Problem Statement
Industrial bins often overflow due to irregular manual checks. This leads to hygiene problems and inefficient waste collection.

**Objective:** Automate the process using sensor data and ML to predict when the bin is full and schedule timely pickups.

---

## 🧪 How It Works

### 🔧 Hardware Setup
- **2 IR Sensors**:
  - **Top sensor:** Detects proximity of waste to the bin lid.
  - **Bottom sensor:** Confirms whether the bin is empty.
- **Microcontroller (e.g., Arduino/ESP32)** reads sensor values and logs them as distance data in cm.
- Data is exported in CSV format for ML training.

### 🤖 Machine Learning Model
- **Input Feature:** `Waste Level (in cm)`
- **Target Label:** `is_full` (1 = full, 0 = not full)
- **Algorithm:** `RandomForestClassifier` from `scikit-learn`
- A bin is considered **full** if the waste is within **3 cm** of the top.

---
