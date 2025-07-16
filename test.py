import os
import logging
from tqdm import tqdm

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def list_directory():
    """List the current directory, its folders, and files, excluding __pycache__."""
    current_dir = os.getcwd()
    logger.info(f"Current directory: {current_dir}")

    # Get list of all entries in the current directory
    entries = []
    for entry in os.scandir(current_dir):
        if entry.name != '__pycache__':
            entries.append(entry)

    # Separate folders and files
    folders = [entry.name for entry in entries if entry.is_dir()]
    files = [entry.name for entry in entries if entry.is_file()]

    # Log folders with progress bar
    logger.info("Folders:")
    if folders:
        for folder in tqdm(folders, desc="Listing folders", unit="folder"):
            logger.info(f"  {folder}")
    else:
        logger.info("  No folders found (excluding __pycache__).")

    # Log files with progress bar
    logger.info("Files:")
    if files:
        for file in tqdm(files, desc="Listing files", unit="file"):
            logger.info(f"  {file}")
    else:
        logger.info("  No files found.")

if __name__ == "__main__":
    list_directory()
    
    