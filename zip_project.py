import zipfile
import os

def zip_project(src_dir, zip_filename):
    exclusions = {'.venv', '.idea', '__pycache__', 'notes.txt', 'notes_video.txt', 'vids', 'old', 'pictures',
                  '46_Calidus0', 'temp.py', 'temp2.py', 'tutorials'}

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for foldername, subfolders, filenames in os.walk(src_dir):

            # Skip excluded folders and files
            if any(ex in foldername for ex in exclusions):
                continue
            for filename in filenames:
                if filename in exclusions:
                    continue

                file_path = os.path.join(foldername, filename)

                # Skip the output zip file itself
                if os.path.abspath(file_path) == os.path.abspath(zip_filename):
                    continue

                arcname = os.path.relpath(file_path, src_dir)
                zipf.write(file_path, arcname)

# Example usage
zip_project('./', './47_Calidus2_Pygame.zip')
