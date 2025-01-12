import os
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers import Dense
from scipy.io.wavfile import write
import librosa

class TextToSpeechModel:
    def __init__(self, model_path=None):
        if model_path:
            self.model = tf.keras.models.load_model(model_path)  # Load the trained model if a path is provided
        else:
            self.model = None  # If no path is provided, initialize an empty model

    def train(self, tsv_filepath, spectrograms_folder):
        """
        Train the model using a .tsv file and a folder of spectrogram PNGs.
        """
        # Step 1: Load the .tsv file (using pandas or any method you prefer)
        import pandas as pd
        data = pd.read_csv(tsv_filepath, sep='\t')
        
        # Step 2: Preprocess data (text and spectrogram paths)
        X_train, y_train = self.preprocess_data(data, spectrograms_folder)
        X_train = tf.keras.utils.pad_sequences(X_train, padding='post', dtype='float32')
        y_train = tf.keras.utils.pad_sequences(y_train, padding='post')

        # Step 3: Build and compile the model
        self.model = self._build_model(X_train.shape[1], y_train.shape[1])

        # Step 4: Train the model
        self.model.fit(X_train, y_train, epochs=20, batch_size=2)

        # Step 5: Save the model
        self.model.save('trained_model.h5')  # Saving the trained model
        print("Model saved successfully!")

    def _build_model(self, input_shape, output_shape):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(output_shape, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


    def preprocess_data(self, data, spectrograms_folder):
        """
        Preprocess data from the .tsv file and convert spectrogram PNGs into arrays.
        """
        X = []  # Input data (spectrograms)
        y = []  # Labels (text)

        for index, row in data.iterrows():
            sentence = row['sentence']
            spectrogram_path = os.path.join(spectrograms_folder, row['path'].replace('.mp3', '_spectrogram.png'))
            
            if not os.path.exists(spectrogram_path):
                print(f"Skipping missing spectrogram: {spectrogram_path}")
                continue  # Skip if file does not exist

            # Load the spectrogram as an array
            spectrogram = self.load_spectrogram(spectrogram_path)
            spectrogram = spectrogram.reshape(-1, 1)  # Reshaping for LSTM input
            
            X.append(spectrogram)
            y.append(self.text_to_labels(sentence))  # Convert text to labels

        return np.array(X, dtype=object), np.array(y, dtype=object)

    def load_spectrogram(self, filepath):
        """
        Load a spectrogram PNG file and return it as an array.
        """
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img = mpimg.imread(filepath)
        return img  # Return the spectrogram as a numpy array

    def text_to_labels(self, text):
        """
        Convert text to labels (this is just a placeholder, you would use tokenization).
        """
        return [ord(c) for c in text]  # Example: convert text to integer labels

    def generate_speech(self, text, output_filepath="generated_audio.wav"):
        """
        Generate speech from text using the trained model and save it to a WAV file.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Please train the model first.")

        # Convert text to labels (you should convert it to the format that works for your model)
        input_data = self.text_to_labels(text)
        input_data = np.array(input_data).reshape(1, -1, 1)

        # Generate the spectrogram
        predicted_spectrogram = self.model.predict(input_data)

        # Convert the spectrogram back to audio (this can involve techniques like Griffin-Lim)
        audio = self.spectrogram_to_audio(predicted_spectrogram)

        # Save audio to WAV
        write(output_filepath, 22050, audio)
        print(f"Generated audio saved to {output_filepath}")

    def spectrogram_to_audio(self, spectrogram):
        """
        Convert the predicted spectrogram back into audio (placeholder method).
        You can use an inverse transform like Griffin-Lim.
        """
        # For simplicity, let's assume this returns a simple sine wave or use a real method
        audio = librosa.griffinlim(spectrogram)
        return audio

    def load_trained_model(self, model_filepath):
        """
        Load a pre-trained model for inference.
        """
        self.model = tf.keras.models.load_model(model_filepath)
