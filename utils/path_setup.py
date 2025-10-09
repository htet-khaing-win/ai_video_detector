#Setting file path
import sys
from pathlib import Path

def add_project_root():
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root