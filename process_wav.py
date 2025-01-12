import os
import argparse
from pydub import AudioSegment
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

def normalize_audio(audio):
    """ Normalize the audio to ensure consistent volume """
    return audio.apply_gain(-audio.dBFS)  # Normalize to -0 dBFS

def resample_audio(audio, target_rate=22050):
    """ Resample the audio to the desired sample rate """
    return audio.set_frame_rate(target_rate)

# def trim_silence(audio, silence_thresh=-40, min_silence_len=1000):
#     """ Trim leading and trailing silence from the audio """
#     return audio.strip_silence(silence_thresh=silence_thresh, min_silence_len=min_silence_len)

def save_spectrogram(wav_file, audio, sr):
    """ Save the Mel spectrogram of the audio """
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram for {wav_file}')
    plt.tight_layout()
    spectrogram_file = wav_file.replace('.wav', '_spectrogram.png')
    plt.savefig(spectrogram_file)
    plt.close()

def process_audio(wav_file, output_dir):
    """ Process a single WAV file: resample, normalize, and trim silence """
    try:
        # Load the audio file
        audio = AudioSegment.from_wav(wav_file)
        
        # Normalize audio
        audio = normalize_audio(audio)
        
        # Resample audio
        audio = resample_audio(audio)
        
        # Trim silence
        # audio = trim_silence(audio)
        
        # Save the processed audio
        output_file = os.path.join(output_dir, os.path.basename(wav_file))
        audio.export(output_file, format="wav")
        print(f"Processed {wav_file} -> {output_file}")

        # Optional: Extract and save the spectrogram (uncomment if needed)
        y, sr = librosa.load(output_file, sr=None)
        save_spectrogram(wav_file, y, sr)
        
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")

def process_directory(directory):
    """ Process all WAV files in the specified directory """
    output_dir = os.path.join(directory, "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            wav_file = os.path.join(directory, filename)
            process_audio(wav_file, output_dir)

def main():
    """ Main function to parse arguments and process the audio files """
    parser = argparse.ArgumentParser(description="Process WAV files in a directory (normalize, resample, trim silence).")
    parser.add_argument('directory', type=str, help="The path to the directory containing WAV files.")
    args = parser.parse_args()

    # Process the WAV files in the provided directory
    process_directory(args.directory)

if __name__ == "__main__":
    main()
