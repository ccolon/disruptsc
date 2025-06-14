import pathlib
import sys
import os
#logger = logging.getLogger(__name__)

ROOT_FOLDER = pathlib.Path(__file__).parent.parent
PARAMETER_FOLDER = ROOT_FOLDER / "parameter"
OUTPUT_FOLDER = ROOT_FOLDER / "output"
TMP_FOLDER = ROOT_FOLDER / "tmp"

# Support configurable data path for repository separation
DATA_PATH = os.environ.get('DISRUPT_SC_DATA_PATH')
if DATA_PATH:
    INPUT_FOLDER = pathlib.Path(DATA_PATH)
elif (ROOT_FOLDER / "data").exists():
    INPUT_FOLDER = ROOT_FOLDER / "data"  # Git submodule
else:
    INPUT_FOLDER = ROOT_FOLDER / "input"  # Fallback to current structure

sys.path.insert(1, str(ROOT_FOLDER))
# if __file__ == "__main__":
#     print(ROOT_FOLDER)

# code_directory = Path(os.path.abspath(__file__)).parent
# project_directory = code_directory.parent
# working_directory = Path(os.getcwd())
# working_directory_parent = working_directory.parent