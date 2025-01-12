import os
import shutil
import argparse

def move_mp3_files(source_folder):
    # Define the target folder
    target_folder = f"{source_folder}MP3"

    # Check if the target folder exists, create it if not
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created directory: {target_folder}")
    
    # Loop through the files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the file is an MP3 file
        if filename.endswith(".mp3"):
            # Construct full file paths
            source_file = os.path.join(source_folder, filename)
            target_file = os.path.join(target_folder, filename)

            # Move the file
            shutil.move(source_file, target_file)
            print(f"Moved {filename} to {target_folder}")

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Move all MP3 files from the source folder to a new folder.")
    
    # Add a required argument for the source folder
    parser.add_argument('source_folder', type=str, help="The path to the source folder containing MP3 files.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Move MP3 files from the provided source folder
    move_mp3_files(args.source_folder)

if __name__ == "__main__":
    main()
