import os
import re
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold  # Add KFold for cross-validation
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report

from ray import tune  # Add Ray Tune for hyperparameter optimization
from torch import nn  # Add for Bidirectional LSTM


file_path = './logss.txt'  # Replace with the path to your txt file

# Read log file
with open(file_path, 'r') as file:
    log_data = file.readlines()

# Regular expression to match error levels (case-insensitive)
error_level_pattern = re.compile(r'\[(crit|error|warn|info|alert|emerg|notice|debug)\]', re.IGNORECASE)

# Extract error levels and handle unmatched cases
error_levels = [match.group(1) for line in log_data if (match := error_level_pattern.search(line))]

# Count occurrences of each error level
error_level_counts = Counter(error_levels)

# Display the counts
print("Error Level Distribution:")
for level, count in error_level_counts.items():
    print(f"{level}: {count}")
    
    
# Initialize an empty list to store the log entries
log_entries = []

# Open and read the log file
with open(file_path, "r") as file:
    for line in file:
        log_entry = line.strip()  # Remove any leading/trailing whitespaces
        log_entries.append(log_entry)

# Define a regex pattern to extract the required components from each log entry
log_pattern = re.compile(r"\[(?P<timestamp>.*?)\] \[(?P<level>\w+)\] \[client (?P<client_ip>\d+\.\d+\.\d+\.\d+)\] (?P<message>.+)")

# Initialize a list to store parsed log details
parsed_logs = []

# Iterate through each log entry and parse it
for log in log_entries:
    match = log_pattern.match(log)
    if match:
        log_details = {
            "timestamp": match.group("timestamp"),
            "level": match.group("level"),
            "client_ip": match.group("client_ip"),
            "message": match.group("message")
        }
        parsed_logs.append(log_details)
    else:
        print(f"Could not parse log entry: {log}")  # Debugging unmatched entries


severity_mapping = {
    "emerg": "High",
    "alert": "High",
    "crit": "High",
    "error": "Medium",
    "warn": "Low",
    "info": "Low",
}

impact_mapping = {
    "emerg": "System Unusable",
    "alert": "Immediate Action Required",
    "crit": "Critical Conditions",
    "error": "Error Condition",
    "warn": "Warning Condition",
    "info": "Informational Condition",
}

# Augment each parsed log with severity and impact ratings
for log in parsed_logs:
    log_level = log["level"].lower()  # Ensure level is lowercase to match the mappings
    log["severity"] = severity_mapping.get(log_level, "Unknown")
    log["impact"] = impact_mapping.get(log_level, "Unknown")
    if log["severity"] == "Unknown" or log["impact"] == "Unknown":
        print(f"Unknown severity or impact for log level: {log_level}")
        
        
# Create a DataFrame from parsed logs for easier manipulation
log_df = pd.DataFrame(parsed_logs)

# Extract relevant features
features_df = log_df[['level', 'message', 'severity', 'impact']]

# Initialize LabelEncoders for the target variables
level_encoder = LabelEncoder()
severity_encoder = LabelEncoder()
impact_encoder = LabelEncoder()

# Fit and transform the target variables safely with .loc
features_df.loc[:, 'level'] = level_encoder.fit_transform(features_df['level'])
features_df.loc[:, 'severity'] = severity_encoder.fit_transform(features_df['severity'])
features_df.loc[:, 'impact'] = impact_encoder.fit_transform(features_df['impact'])


# Initialize the TF-IDF Vectorizer with stop words and bi-grams
tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))

# Fit and transform the error messages to create TF-IDF features
tfidf_features = tfidf_vectorizer.fit_transform(features_df['message']).toarray()

# Create a DataFrame for TF-IDF features
tfidf_features_df = pd.DataFrame(tfidf_features, columns=tfidf_vectorizer.get_feature_names_out())

# Combine TF-IDF features with the existing features DataFrame
final_features_df = pd.concat([features_df.reset_index(drop=True), tfidf_features_df.reset_index(drop=True)], axis=1)


# Initialize the BERT Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Check if a GPU is available and move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to generate BERT embeddings with batching and GPU support
def get_bert_embeddings(messages, batch_size=32):
    embeddings = []
    for i in range(0, len(messages), batch_size):
        batch_messages = messages[i:i+batch_size]
        inputs = tokenizer(batch_messages, return_tensors='pt', truncation=True, padding=True, max_length=128,
                           add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False)

        # Move inputs to the GPU if available
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Move embeddings back to CPU for further processing
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(sentence_embedding)

    return np.vstack(embeddings)

# Generate BERT embeddings for the error messages
bert_embeddings = get_bert_embeddings(features_df['message'].tolist())

# Create a DataFrame for BERT embeddings
bert_df = pd.DataFrame(bert_embeddings)

# Combine BERT embeddings with the existing features DataFrame
final_features_df = pd.concat([features_df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)



# Importing required libraries
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch
import joblib
import numpy as np

# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)  # Fully connected layer for classification

    def forward(self, x):
        # Reshape the input to (batch_size, sequence_length=1, input_size)
        x = x.unsqueeze(1)  # Add a dimension for sequence_length
        out, (hn, _) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Taking the output from the last timestep
        return out

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()

    return total_loss / len(train_loader)

# Function to evaluate the model
def evaluate_model(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    return correct / total

# Prepare the data (assuming tfidf_features are used for LSTM)
X = tfidf_features_df.values  # Use TF-IDF features or raw tokenized sequences

# Function to train, save models, and evaluate
def train_and_save_model(target_variable, model_save_path, encoder_save_path):
    y = features_df[target_variable].values
    if not np.issubdtype(y.dtype, np.integer):  # Check if 'y' contains non-integer values
        print("Encoding target labels...")
        y = level_encoder.fit_transform(y)  # Re-apply LabelEncoder if needed

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Create a dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # KFold Cross Validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    input_size = X_tensor.shape[1]  # Number of features (from TF-IDF or embeddings)
    hidden_size = 128  # Number of units in the LSTM
    num_layers = 2  # Number of LSTM layers
    num_classes = len(np.unique(y))  # Number of output classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # KFold Loop
    for fold, (train_index, val_index) in enumerate(kf.split(X_tensor)):
        print(f"Fold {fold + 1}/{n_splits}")

        # Split data into training and validation sets
        X_train, X_val = X_tensor[train_index], X_tensor[val_index]
        y_train, y_val = y_tensor[train_index], y_tensor[val_index]

        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

        # Initialize the model, loss function, and optimizer
        model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001)

        # Train the model for a number of epochs
        num_epochs = 5
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_accuracy = evaluate_model(model, val_loader, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save the model after training for each fold
        torch.save(model.state_dict(), f"{model_save_path}_{target_variable}_fold_{fold + 1}.pth")

    # Save the encoder for the target variable
    joblib.dump(level_encoder, encoder_save_path)
    print(f"Model for {target_variable} saved.")

# Train and save models for each target variable
train_and_save_model('level', 'lstm_classifier', 'level_encoder.pkl')
train_and_save_model('severity', 'lstm_classifier', 'severity_encoder.pkl')
train_and_save_model('impact', 'lstm_classifier', 'impact_encoder.pkl')

def predict_error_message(models, error_message):
    predictions = {}

    # Transform the message to the feature space
    transformed_message = tfidf_vectorizer.transform([error_message]).toarray()
    message_tensor = torch.tensor(transformed_message, dtype=torch.float32).to(device)

    # Iterate through each model and make predictions
    for target, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(message_tensor)
            _, predicted = torch.max(output.data, 1)

            if target == 'level':
                predictions[target] = level_encoder.inverse_transform(predicted.cpu().numpy())[0]
            elif target == 'severity':
                predictions[target] = severity_encoder.inverse_transform(predicted.cpu().numpy())[0]
            elif target == 'impact':
                predictions[target] = impact_encoder.inverse_transform(predicted.cpu().numpy())[0]

    return predictions


# Function to load models for prediction (ensure to create a separate LSTMClassifier class for each target variable)
def load_model(model_path, input_size, hidden_size, num_layers, num_classes):
    model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    # Move the loaded model to the appropriate device (GPU if available)
    model.to(device)  # Add this line
    return model

# Function to load the appropriate model and its number of classes
def load_models_for_prediction():
    models = {}
    # Define model configurations with input_size, hidden_size, num_layers
    model_configs = {
        'level': ('lstm_classifier_level_fold_1.pth', len(level_encoder.classes_), X.shape[1], 128, 2), # Add input_size, hidden_size, num_layers
        'severity': ('lstm_classifier_severity_fold_1.pth', len(severity_encoder.classes_), X.shape[1], 128, 2),  # Add input_size, hidden_size, num_layers
        'impact': ('lstm_classifier_impact_fold_1.pth', len(impact_encoder.classes_), X.shape[1], 128, 2),  # Add input_size, hidden_size, num_layers
    }

    for target, (model_path, num_classes, input_size, hidden_size, num_layers) in model_configs.items(): # Unpack input_size, hidden_size, num_layers
        models[target] = load_model(model_path, input_size, hidden_size, num_layers, num_classes)

    return models

# Example usage
error_message = "[Fri Aug 17 14:29:51 2007] [warn] [client 192.168.1.99] Module not found: 'mod_rewrite'"
models = load_models_for_prediction()



from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)



@app.route('/log', methods=['POST'])
def log_error():
    error_log = request.json.get('error_log')
    if not error_log:
        return jsonify({'error': 'No error log provided'}), 400

    # Process the error log here
    # You can call your existing error handling function or manage the log as needed
    output = predict_error_message(models, error_log) # Assuming process_error_log is a function defined elsewhere
    return jsonify({'message': 'Error log processed', 'result': output})

# Start the Flask app in a thread
def run_flask():
    app.run(host='0.0.0.0', port=3002)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

# Set up ngrok to expose the Flask app

print(f"Flask app is running at: 3002")
