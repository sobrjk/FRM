import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import librosa

# Function to extract features from an audio file
def extract_features(file_path):
    sample_rate, audio_data = wavfile.read(file_path)
    # Convert to mono if stereo
    if len(audio_data.shape) == 2:
        audio_data = audio_data.mean(axis=1)
    
    # Features might include spectrogram, Mel-frequency cepstral coefficients, etc.
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Data loading
# Assume 'X' is a list of audio file paths, 'y' is the labels (1 for fart, 0 for non-fart)
X = [...]
y = [...]

# Extract features for all files
features = [extract_features(file) for file in X]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Train the model (using RandomForest as an example)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Function to predict if an audio is a fart
def predict_fart(file_path):
    features = extract_features(file_path)
    prediction = model.predict([features])
    return "It's a fart!" if prediction[0] == 1 else "It's not a fart."

# Example usage
# predict_fart('path_to_new_sound.wav')
