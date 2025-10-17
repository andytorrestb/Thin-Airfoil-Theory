import sys
from pathlib import Path

# Ensure the project's src directory is on sys.path so tests can import the package modules
HERE = Path(__file__).resolve().parent
SRC = (HERE / '..' / 'src').resolve()
sys.path.insert(0, str(SRC))