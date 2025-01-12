from pydub import AudioSegment
import os
import argparse

def mp3_to_wav(mp3_file, wav_file):
    try:
        audio = AudioSegment.from_mp3(mp3_file)
        audio.export(wav_file, format="wav")
        print(f"Successfully converted {mp3_file} to {wav_file}")
    except Exception as e:
        print(f"Error: {e}")
        
def convert_all_mp3_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp3"):
            mp3_path = os.path.join(directory, filename)
            wav_path = os.path.join(directory, filename.replace(".mp3", ".wav"))
            mp3_to_wav(mp3_path, wav_path)

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Convert MP3 files to WAV in a specified directory.")
    
    # Add a required argument for the directory path
    parser.add_argument('directory', type=str, help="The path to the directory containing MP3 files.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Convert MP3 files in the specified directory
    convert_all_mp3_in_directory(args.directory)

if __name__ == "__main__":
    main()