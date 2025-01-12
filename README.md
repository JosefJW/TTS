# Text-to-Speech (TTS)

## Overview

This Text-to-Speech (TTS) project aims to build a machine learning model that converts textual input into corresponding audio output. The project uses an LSTM-based model to generate spectrograms from text, which are then converted back into audio.

## Features

Train an LSTM-based model on text-spectrogram pairs.

Generate spectrograms from text and convert them into audio.

Support for custom datasets via .tsv files and spectrogram images.

Save and load trained models for inference.

## Project Structure

<pre>
TTS/  
|-- model.py            # Contains the TextToSpeechModel class  
|-- move_mp3_files.py   # Moves mp3 files into a dedicated folder  
|-- mp3_to_wav.py       # Converts MP3 files into WAV files  
|-- process_tsv.py      # Gets filepath and sentence data from a tsv file  
|-- process_wav.py      # Processes WAV files and makes spectrograms  
|-- requirements.txt    # Dependencies for the project  
|-- README.md           # This file  
</pre>

## Requirements

- Python 3.8+

- TensorFlow 2.x

- NumPy

- SciPy

- Librosa

- Matplotlib

- Pandas

<br>

To install the dependencies, run:

pip install -r requirements.txt

## Data Preparation

The model requires:

- A .tsv file containing two columns: sentence (text input) and path (relative path to the corresponding spectrogram image).

- A folder containing spectrogram PNG images corresponding to the path column in the .tsv file.

Example .tsv file:

sentence	path  
Hello world	file1_spectrogram.png  
How are you?	file2_spectrogram.png  

## Training the Model

To train the model:

1. Ensure the .tsv file and spectrogram images are properly prepared.

2. Make an instance of the model class.

3. Pass the model the filepaths in model.train(tsv_filepath, spectrograms_folderpath).

The trained model will be saved as trained_model.h5 upon completion.

## Model Architecture

The model consists of:

- LSTM layer for sequential data processing.

- TimeDistributed Dense layer for intermediate processing.

- Dense layer with softmax activation for output.

## Generating Speech

After training, use the generate_speech method to convert text into audio:

from model import TextToSpeechModel

Load the trained model  
tts_model = TextToSpeechModel(model_path='trained_model.h5')

Generate audio  
tts_model.generate_speech("Hello world", output_filepath="output.wav")

## Future Improvements

- Implement more efficient data preprocessing.

- Explore alternative architectures for improved performance.

- Add support for different audio synthesis techniques.

## License

This project is open-source and available under the MIT License.

## Acknowledgments

TensorFlow and Keras for providing a robust ML framework.

Librosa and SciPy for audio processing utilities.

Feel free to contribute to the project by submitting pull requests or reporting issues!
