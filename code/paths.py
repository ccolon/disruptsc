import pathlib

#logger = logging.getLogger(__name__)

ROOT_FOLDER = pathlib.Path(__file__).parent.parent

if __file__ == "__main__":
    print(ROOT_FOLDER)

# code_directory = Path(os.path.abspath(__file__)).parent
# project_directory = code_directory.parent
# working_directory = Path(os.getcwd())
# working_directory_parent = working_directory.parent