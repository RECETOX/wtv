from pathlib import Path
import glob

# Get the directory of the current file
current_dir = Path(__file__).resolve().parent

# Dictionary to store file paths
test_files = {}

# List of valid extensions
valid_extensions = [".csv", ".msp"]

# List all files in the directory and subdirectories
for file in glob.glob(f"{current_dir}/**/*", recursive=True):
    if Path(file).suffix in valid_extensions:
        file_name = Path(file).stem
        test_files[file_name] = file


def get_test_data_base_path():
    return str(current_dir)


def get_test_file(file_name, default=None):

    return test_files.get(file_name, default)


# Export the test_files dictionary and functions for easy import
__all__ = ["get_test_data_base_path", "get_test_file"]
