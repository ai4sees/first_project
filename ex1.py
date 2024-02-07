import os
import shutil
import random
https://drive.google.com/drive/u/1/folders/1LIBrF7P3UirrPUj6ZhIlfbXxv4Iq4oib

def pick_and_copy_images(src_folder, dest_folder, percentage=0.2):
    # Ensure the destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # List all files in the source folder
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    
    # Shuffle the list of files
    random.shuffle(files)

    # Calculate the number of files to select
    num_files_to_select = int(len(files) * percentage)

    # Select the files
    selected_files = files[:num_files_to_select]

    # Copy the selected files to the destination folder
    for file in selected_files:
        shutil.copy(os.path.join(src_folder, file), os.path.join(dest_folder, file))

# Example usage
src_folder = 'path/to/source/folder'
dest_folder = 'path/to/destination/folder'
pick_and_copy_images(src_folder, dest_folder, percentage=0.2)
