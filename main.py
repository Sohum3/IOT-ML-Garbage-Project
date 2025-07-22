import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Configuration ---
MODEL_PATH = 'waste_management_model.pkl'
TRAINING_DATA_PATH = 'waste_data.csv'
TESTING_DATA_PATH = 'waste_testing_data.csv'
PREDICTIONS_PATH = 'waste_predictions.csv'
CONFIDENCE_THRESHOLD = 0.95 # Use predictions with 95% or greater confidence as new training data

def train_and_save_model(data_path, model_path):
    """Trains a RandomForestClassifier and saves it to a file."""
    print(f"Loading training data from {data_path}...")
    df = pd.read_csv(data_path)

    if df.shape[0] < 2:
        print("Not enough data to train a model. Need at least 2 samples.")
        return None

    X = df[['waste_level_cm']]
    y = df['is_full']

    # Initialize and train the model
    # n_estimators and random_state are set for reproducibility
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("Model training complete.")

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}.")
    return model

def main():
    """
    Main function to run the prediction and active learning pipeline.
    """
    print("--- Starting Smart Waste Management Process ---")

    # --- Step 1: Initial Model Training (if no model exists) ---
    if not os.path.exists(MODEL_PATH):
        print(f"No existing model found at {MODEL_PATH}. Performing initial training.")
        train_and_save_model(TRAINING_DATA_PATH, MODEL_PATH)

    # --- Step 2: Load Model and New Data ---
    print("\n--- Loading existing model and new sensor data ---")
    try:
        model = joblib.load(MODEL_PATH)
        new_data = pd.read_csv(TESTING_DATA_PATH)
        print(f"Loaded model from {MODEL_PATH}.")
        print(f"Loaded {len(new_data)} new sensor readings from {TESTING_DATA_PATH}.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # --- Step 3: Make Predictions on New Data ---
    print("\n--- Generating predictions on new data ---")
    X_new = new_data[['waste_level_cm']]
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)

    # Add predictions and confidence to the dataframe
    new_data['is_full_predicted'] = predictions
    new_data['confidence'] = np.max(probabilities, axis=1)

    # Save the predictions to a CSV file
    new_data.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}.")
    print("Predictions with confidence scores:")
    print(new_data)


    # --- Step 4: Active Learning - Identify and Add High-Confidence Samples ---
    print("\n--- Starting Active Learning Loop ---")
    
    # Filter for samples that meet the confidence threshold
    high_confidence_samples = new_data[new_data['confidence'] >= CONFIDENCE_THRESHOLD].copy()

    if not high_confidence_samples.empty:
        print(f"Found {len(high_confidence_samples)} new samples with >= {CONFIDENCE_THRESHOLD*100}% confidence.")
        
        # Prepare the new data for training
        new_training_data = high_confidence_samples[['waste_level_cm', 'is_full_predicted']]
        new_training_data.rename(columns={'is_full_predicted': 'is_full'}, inplace=True)
        
        # Append to the main training data file without writing the header
        new_training_data.to_csv(TRAINING_DATA_PATH, mode='a', header=False, index=False)
        print(f"Appended new samples to {TRAINING_DATA_PATH}.")

        # --- Step 5: Retrain the Model with Augmented Data ---
        print("\n--- Retraining model with augmented dataset ---")
        train_and_save_model(TRAINING_DATA_PATH, MODEL_PATH)
        print("Model has been updated with new knowledge.")

    else:
        print("No new samples met the confidence threshold. Model was not retrained.")

    print("\n--- Smart Waste Management Process Complete ---")

if __name__ == "__main__":
    main()
