import pathlib
import sys
import os
import time
import shutil
#logger = logging.getLogger(__name__)

ROOT_FOLDER = pathlib.Path(__file__).parent.parent.parent
PARAMETER_FOLDER = ROOT_FOLDER / "config" / "parameters"
OUTPUT_FOLDER = ROOT_FOLDER / "output"
TMP_FOLDER = ROOT_FOLDER / "tmp"

# Global variable to store the isolated cache directory for this process
_ISOLATED_CACHE_DIR = None
_CACHE_ISOLATION_ENABLED = False

def setup_cache_isolation(scope: str):
    """Setup isolated cache directory for this process."""
    global _ISOLATED_CACHE_DIR, _CACHE_ISOLATION_ENABLED
    
    if _CACHE_ISOLATION_ENABLED:
        return  # Already setup
    
    process_id = os.getpid()
    timestamp = round(time.time() * 1000)
    dir_name = f"{scope}_pid_{process_id}_{timestamp}"
    
    _ISOLATED_CACHE_DIR = TMP_FOLDER / dir_name
    _ISOLATED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_ISOLATION_ENABLED = True

def get_cache_dir() -> pathlib.Path:
    """Get the cache directory (isolated or shared)."""
    if _CACHE_ISOLATION_ENABLED and _ISOLATED_CACHE_DIR:
        return _ISOLATED_CACHE_DIR
    return TMP_FOLDER

def cleanup_isolated_cache():
    """Clean up isolated cache directory."""
    global _ISOLATED_CACHE_DIR, _CACHE_ISOLATION_ENABLED
    
    if _CACHE_ISOLATION_ENABLED and _ISOLATED_CACHE_DIR and _ISOLATED_CACHE_DIR.exists():
        shutil.rmtree(_ISOLATED_CACHE_DIR)
        _ISOLATED_CACHE_DIR = None
        _CACHE_ISOLATION_ENABLED = False

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