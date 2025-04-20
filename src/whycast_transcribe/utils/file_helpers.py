import os
import logging
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm
from whycast_transcribe.transcribe import write_transcript_files  # Expose from main module


def check_file_size(file_path: str, max_file_size_kb: int) -> bool:
    size_kb = os.path.getsize(file_path) / 1024
    if size_kb > max_file_size_kb:
        logging.warning(f"File size ({size_kb:.1f} KB) exceeds recommended limit ({max_file_size_kb} KB). Processing might be slow or fail.")
        return False
    return True


def create_output_paths(input_file: str) -> Tuple[str, str, str]:
    base = os.path.splitext(input_file)[0]
    return (
        f"{base}.txt",
        f"{base}_ts.txt",
        f"{base}_summary.txt",
    )


def setup_output_base(input_file: str, output_dir: Optional[str] = None) -> str:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(os.path.splitext(input_file)[0])
        return os.path.join(output_dir, base_name)
    return os.path.splitext(input_file)[0]


def read_prompt_file(file_path: str) -> Optional[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading prompt file {file_path}: {e}")
        return None