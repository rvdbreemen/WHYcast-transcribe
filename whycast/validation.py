"""
Path and file validation for the WHYcast pipeline (ADR-008).

Functions extracted verbatim from transcribe.py as part of the mechanical
monolith split (ADR-008): validate_file_path, validate_directory_path,
check_file_size.
"""

import logging
import os

from whycast.config import MAX_FILE_SIZE_KB
from whycast.errors import SecurityError

logger = logging.getLogger(__name__)


def validate_file_path(file_path: str, must_exist: bool = True, check_readable: bool = True) -> str:
    """
    Validate and sanitize a file path for security.
    
    Args:
        file_path: The file path to validate
        must_exist: Whether the file must exist
        check_readable: Whether to check if the file is readable
        
    Returns:
        The validated and normalized file path
        
    Raises:
        ValueError: If the file path is invalid
        SecurityError: If the file path contains suspicious elements
        FileNotFoundError: If the file must exist but doesn't
    """
    if not file_path or not isinstance(file_path, str):
        raise ValueError("File path must be a non-empty string")
    
    # Security: Remove any null bytes
    file_path = file_path.replace('\0', '')
    
    # Security: Check for path traversal attempts
    if '..' in file_path or file_path.startswith('/') or '\\' in file_path.replace(os.sep, ''):
        # Allow normal Windows paths but block suspicious ones
        normalized = os.path.normpath(file_path)
        if '..' in normalized:
            raise SecurityError(f"Path traversal detected in file path: {file_path}")
    
    # Security: Check for suspicious characters
    suspicious_chars = ['<', '>', '|', '*', '?', '"']
    if any(char in file_path for char in suspicious_chars):
        raise SecurityError(f"Suspicious characters detected in file path: {file_path}")
    
    # Normalize the path
    normalized_path = os.path.normpath(os.path.abspath(file_path))
    
    # Check existence if required
    if must_exist and not os.path.exists(normalized_path):
        raise FileNotFoundError(f"File not found: {normalized_path}")
    
    # Check if it's actually a file (not a directory) if it exists
    if os.path.exists(normalized_path) and not os.path.isfile(normalized_path):
        raise ValueError(f"Path exists but is not a file: {normalized_path}")
    
    # Check readability if required
    if check_readable and os.path.exists(normalized_path):
        if not os.access(normalized_path, os.R_OK):
            raise PermissionError(f"File is not readable: {normalized_path}")
    
    return normalized_path

def validate_directory_path(dir_path: str, create_if_missing: bool = False) -> str:
    """
    Validate and sanitize a directory path for security.
    
    Args:
        dir_path: The directory path to validate
        create_if_missing: Whether to create the directory if it doesn't exist
        
    Returns:
        The validated and normalized directory path
        
    Raises:
        ValueError: If the directory path is invalid
        SecurityError: If the directory path contains suspicious elements
    """
    if not dir_path or not isinstance(dir_path, str):
        raise ValueError("Directory path must be a non-empty string")
    
    # Security: Remove any null bytes
    dir_path = dir_path.replace('\0', '')
    
    # Security: Check for path traversal attempts
    if '..' in dir_path:
        normalized = os.path.normpath(dir_path)
        if '..' in normalized:
            raise SecurityError(f"Path traversal detected in directory path: {dir_path}")
    
    # Security: Check for suspicious characters
    suspicious_chars = ['<', '>', '|', '*', '?', '"']
    if any(char in dir_path for char in suspicious_chars):
        raise SecurityError(f"Suspicious characters detected in directory path: {dir_path}")
    
    # Normalize the path
    normalized_path = os.path.normpath(os.path.abspath(dir_path))
    
    # Create directory if requested and it doesn't exist
    if create_if_missing and not os.path.exists(normalized_path):
        try:
            os.makedirs(normalized_path, exist_ok=True)
            logging.info(f"Created directory: {normalized_path}")
        except Exception as e:
            raise PermissionError(f"Cannot create directory {normalized_path}: {str(e)}")
    
    # Check if it's actually a directory if it exists
    if os.path.exists(normalized_path) and not os.path.isdir(normalized_path):
        raise ValueError(f"Path exists but is not a directory: {normalized_path}")
    
    return normalized_path

def check_file_size(file_path: str) -> bool:
    """
    Check if the file size is within acceptable limits.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if file size is acceptable, False if it's too large
    """
    size_kb = os.path.getsize(file_path) / 1024
    if size_kb > MAX_FILE_SIZE_KB:
        logging.warning(f"File size ({size_kb:.1f} KB) exceeds recommended limit ({MAX_FILE_SIZE_KB} KB). " 
                      f"Processing might be slow or fail.")
        return False
    return True
