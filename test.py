import os

def print_directory_structure(root_dir):
    """
    Print directory structure starting from root_dir.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Calculate indent based on depth
        depth = dirpath.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * depth
        print(f"{indent}{os.path.basename(dirpath)}/")
        
        # Print files in the current directory
        for file in filenames:
            print(f"{indent}    {file}")

if __name__ == "__main__":
    current_dir = os.getcwd()  # Get current working directory
    print(f"Current Directory: {current_dir}\n")
    print("Directory Structure:\n")
    print_directory_structure(current_dir)
