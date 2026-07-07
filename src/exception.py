import sys
import os
import logging

# Make sure `src` package (and `src/logger.py`) is importable when running
# the script directly (python src/exception.py). This inserts the `src`
# directory on sys.path so `import logger` finds `src/logger.py`.
project_root = os.getcwd()
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    import logger  # noqa: F401 - configures logging
except Exception:
    # If importing fails, rely on the default logging configuration
    pass

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message  

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
