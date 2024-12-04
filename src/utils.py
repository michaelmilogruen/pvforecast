import os

def get_project_root():
    """Returns project root folder."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_path(filename, data_type='raw'):
    """
    Returns full path for data files.
    Args:
        filename: Name of the file
        data_type: Either 'raw' or 'processed'
    """
    return os.path.join(get_project_root(), 'data', data_type, filename)

def get_output_path(filename, output_type='figures'):
    """
    Returns full path for output files.
    Args:
        filename: Name of the file
        output_type: Type of output (e.g., 'figures')
    """
    return os.path.join(get_project_root(), 'outputs', output_type, filename)

def get_model_path(filename):
    """Returns full path for model files."""
    return os.path.join(get_project_root(), 'models', filename) 