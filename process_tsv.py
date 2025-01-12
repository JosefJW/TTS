import pandas as pd
import argparse

# Function to modify the audio path to end with _spectrogram.png
def modify_audio_path(audio_path):
    if audio_path.endswith('.mp3'):
        # Replace .mp3 with _spectrogram.png
        return audio_path.replace('.mp3', '_spectrogram.png')
    return audio_path  # In case the path does not end with .mp3, return the original

# Function to process the .tsv file
def process_tsv(tsv_file):
    # Load the TSV file into a DataFrame
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Process each row: modify the path
    for index, row in df.iterrows():
        mp3_path = row['path']
        
        # Modify the path from .mp3 to _spectrogram.png
        updated_path = modify_audio_path(mp3_path)
        
        # Update the path in the DataFrame
        df.at[index, 'path'] = updated_path
    
    # Save the updated DataFrame to a new TSV file
    updated_file = tsv_file.replace('.tsv', '_updated.tsv')
    df.to_csv(updated_file, sep='\t', index=False)
    print(f"Updated TSV file saved as {updated_file}")

def main():
    # Parse the command line argument for the .tsv file path
    parser = argparse.ArgumentParser(description="Update audio paths in a TSV file to point to spectrograms.")
    parser.add_argument('tsv_file', type=str, help="Path to the TSV file.")
    args = parser.parse_args()

    # Process the .tsv file
    process_tsv(args.tsv_file)

if __name__ == "__main__":
    main()
