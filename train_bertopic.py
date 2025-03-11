import logging
import pandas as pd
from bertopic import BERTopic
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Data Loading
def load_data(file_path):
    if not Path(file_path).exists():
        logging.error(f"Data file not found: {file_path}")
        exit(1)
    try:
        data = pd.read_csv(file_path)
        if 'text' not in data.columns:
            logging.error("Error: 'text' column not found in data.")
            exit(1)
        logging.info(f"Data loaded successfully with {len(data)} rows.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        exit(1)

# BERTopic Model Training
def train_model(text_data):
    try:
        topic_model = BERTopic(verbose=True)
        logging.info("Training the BERTopic model...")
        topics, _ = topic_model.fit_transform(text_data)
        logging.info(f"Training complete with {len(set(topics))} topics identified.")
        return topic_model
    except Exception as e:
        logging.error(f"Error during training: {e}")
        exit(1)

# Model Saving
def save_model(model, model_path):
    try:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path, serialization='safetensors')
        logging.info(f"Model saved successfully at '{model_path}'.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

if __name__ == "__main__":
    DATA_PATH = "data/Tweets.csv"  # Update path if needed
    MODEL_PATH = "models/bertopic_model"

    data = load_data(DATA_PATH)
    text_data = data['text'].dropna().tolist()

    if not text_data:
        logging.error("No valid text data found in the dataset.")
        exit(1)

    model = train_model(text_data)
    save_model(model, MODEL_PATH)

    logging.info("All tasks completed successfully!")
