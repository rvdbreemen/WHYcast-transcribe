"""
Logging setup for the WHYcast pipeline (ADR-008).

Extracted verbatim from transcribe.py (setup_logging, lines 144-252) during
the ADR-008 library extraction. The function returns a configured logger; it
is called explicitly by the CLI shim and the web worker, never at import time.

One deliberate deviation: the log file path is anchored to the repository
root via whycast.config.base_dir instead of os.path.dirname(__file__), so
transcribe.log keeps landing in the repo root as it did before extraction.
"""

import logging
import os
import sys

from whycast.config import base_dir


#: torch loggers that are chatty at INFO and say nothing an operator needs.
_NOISY_TORCH_LOGGERS = (
    "torch._inductor",
    "torch._dynamo",
    "torch._functorch",
    "torch._subclasses",
)


def quiet_torch_logging() -> None:
    """Turn down torch's inductor and dynamo chatter.

    Call this *after* ``import torch``. Importing torch configures its own
    logging, which resets whatever levels were set before, so a call from
    :func:`setup_logging` only sticks when torch was already imported - which
    is how it worked in the pre-extraction monolith, where ``import torch`` sat
    at the top of the file and ``setup_logging()`` ran below it. In the library
    the runner calls ``setup_logging()`` first, so the modules that import torch
    call this again afterwards.

    Idempotent, and safe to call when torch is not installed at all.
    """
    for name in _NOISY_TORCH_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


# Set up logging to both console and file
def setup_logging():
    # Enhanced log format with line numbers and function names
    log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Default level set to INFO

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler with UTF-8 encoding
    # Anchored to the repo root via whycast.config.base_dir: in transcribe.py
    # __file__ was the repo root, but after extraction it would resolve to
    # whycast/, silently moving transcribe.log.
    log_file = os.path.join(base_dir, 'transcribe.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.setLevel(logging.INFO)  # Zet file handler op INFO niveau
    logger.addHandler(file_handler)

    # Console handler with proper Unicode handling
    try:        # Configure console for UTF-8
        if sys.platform == 'win32':
            # Use error handler that replaces problematic characters
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            console_handler.setLevel(logging.INFO)  # Zet console handler op INFO niveau
            console_handler.setStream(open(os.devnull, 'w', encoding='utf-8'))  # Dummy stream for initial setup

            # Custom StreamHandler that handles encoding errors
            class EncodingSafeStreamHandler(logging.StreamHandler):
                def emit(self, record):
                    try:
                        msg = self.format(record)
                        stream = self.stream
                        # Write with error handling for encoding issues
                        try:
                            stream.write(msg + self.terminator)
                        except UnicodeEncodeError:
                            # Fall back to ascii with replacement characters
                            stream.write(msg.encode('ascii', 'replace').decode('ascii') + self.terminator)
                        self.flush()
                    except Exception:
                        self.handleError(record)

            # Use our custom handler
            console_handler = EncodingSafeStreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(log_format))
            console_handler.setLevel(logging.INFO)  # Zet custom handler op INFO niveau
        else:
            # On non-Windows platforms, standard handler usually works fine
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(log_format))
            console_handler.setLevel(logging.INFO)  # Zet console handler op INFO niveau

        logger.addHandler(console_handler)
    except Exception as e:
        # Fallback to basic handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        console_handler.setLevel(logging.INFO)  # Zet fallback handler op INFO niveau
        logger.addHandler(console_handler)
        logger.warning(f"Could not set up optimal console logging: {e}")

    # Zet specifieke loggers die verbose kunnen zijn op WARNING niveau
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Suppress PyTorch Inductor messages
    logging.getLogger("torch._inductor").setLevel(logging.CRITICAL)
    logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)
    logging.getLogger("torch._subclasses").setLevel(logging.CRITICAL)

    # Additional PyTorch logging suppressions
    logging.getLogger("torch._inductor.remote_cache").setLevel(logging.CRITICAL)
    logging.getLogger("torch._dynamo.eval_frame").setLevel(logging.CRITICAL)
    logging.getLogger("torch._dynamo.utils").setLevel(logging.CRITICAL)

    # Suppress PyAnnote Audio warnings
    logging.getLogger("pyannote").setLevel(logging.ERROR)

    # Filter specific warnings types
    import warnings
    warnings.filterwarnings("ignore", message="TensorFloat-32 .* has been disabled")
    warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")
    warnings.filterwarnings("ignore", message=".*Cache Metrics.*")

    # Quieten torch's inductor and dynamo chatter.
    #
    # This used to be os.environ["TORCH_LOGS"] = "ERROR", which torch rejects:
    # that variable takes module names ("dynamo", "inductor"), never log level
    # names, and torch raises ValueError while parsing it. In the pre-extraction
    # monolith the assignment was dead code - setup_logging() ran at import time
    # *after* `import torch`, so torch had already read the variable and never
    # looked again. Extracting the function verbatim (ADR-008) kept the line but
    # changed when it runs: the job runner calls setup_logging() before the
    # pipeline imports torch, so torch finally parsed it and every GPU job died
    # at startup with "Invalid log settings: ERROR".
    #
    # The intent is reachable through Python's own logging, which is where these
    # messages come from anyway.
    quiet_torch_logging()
    os.environ["TORCH_INDUCTOR_VERBOSE"] = "0"

    # Suppress PyTorch Inductor compile_threads warnings
    import logging as py_logging
    class PyTorchFilter(py_logging.Filter):
        def filter(self, record):
            # Filter out the compile_threads messages
            return not (hasattr(record, 'msg') and
                       isinstance(record.msg, str) and
                       "compile_threads set to" in record.msg)

    # Apply the filter to all loggers
    root_logger = py_logging.getLogger()
    root_logger.addFilter(PyTorchFilter())

    return logger
