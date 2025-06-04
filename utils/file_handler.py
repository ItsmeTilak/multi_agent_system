"""
File handling utilities for the multi-agent system.
"""
import os
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO
from datetime import datetime
import mimetypes

from config import FILE_CONFIG, UPLOADS_DIR

logger = logging.getLogger(__name__)


class FileHandler:
    """
    Handles file upload, validation, and management operations.
    """
    
    def __init__(self, upload_dir: Path = UPLOADS_DIR):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(exist_ok=True)
        self.max_size_bytes = FILE_CONFIG["max_file_size_mb"] * 1024 * 1024
        self.allowed_extensions = FILE_CONFIG["allowed_extensions"]
    
    def validate_file(self, file_path: Path, file_type: str = None) -> Dict[str, Any]:
        """
        Validate uploaded file against size and type constraints.
        
        Args:
            file_path: Path to the file to validate
            file_type: Expected file type ('pdf', 'json', 'email')
        
        Returns:
            Dict with validation results
        """
        validation_result = {
            "valid": False,
            "file_size": 0,
            "file_type": None,
            "format": None,  # Add format key
            "mime_type": None,
            "errors": []
        }
        
        try:
            if not file_path.exists():
                validation_result["errors"].append("File does not exist")
                return validation_result
            
            # Check file size
            file_size = file_path.stat().st_size
            validation_result["file_size"] = file_size
            
            if file_size > self.max_size_bytes:
                validation_result["errors"].append(
                    f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds limit "
                    f"({FILE_CONFIG['max_file_size_mb']}MB)"
                )
            
            if file_size == 0:
                validation_result["errors"].append("File is empty")
            
            # Check file extension
            file_extension = file_path.suffix.lower()
            detected_type = None
            
            for type_name, extensions in self.allowed_extensions.items():
                if file_extension in extensions:
                    detected_type = type_name
                    break
            
            if not detected_type:
                validation_result["errors"].append(
                    f"File extension '{file_extension}' not allowed"
                )
            else:
                validation_result["file_type"] = detected_type
                validation_result["format"] = detected_type  # Add format key
            
            # Validate against expected type if provided
            if file_type and detected_type and detected_type != file_type:
                validation_result["errors"].append(
                    f"Expected {file_type} file, but detected {detected_type}"
                )
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            validation_result["mime_type"] = mime_type
            
            # File is valid if no errors
            validation_result["valid"] = len(validation_result["errors"]) == 0
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            validation_result["errors"].append(f"Validation error: {str(e)}")
            return validation_result
    
    def save_uploaded_file(self, file_content: bytes, filename: str, 
                      file_type: str = None) -> Dict[str, Any]:
        """
        Save uploaded file content to the uploads directory.
        
        Args:
            file_content: Raw file content as bytes
            filename: Original filename
            file_type: Expected file type for validation
        
        Returns:
            Dict with save results including new file path
        """
        try:
            # Generate unique filename to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = hashlib.md5(file_content if isinstance(file_content, bytes) else file_content.encode()).hexdigest()[:8]
            name_part, ext_part = os.path.splitext(filename)
            
            safe_filename = f"{name_part}_{timestamp}_{file_hash}{ext_part}"
            file_path = self.upload_dir / safe_filename
            
            # Write file content - FIXED VERSION
            with open(file_path, 'wb') as f:
                if isinstance(file_content, str):
                    f.write(file_content.encode('utf-8'))
                elif isinstance(file_content, bytes):
                    f.write(file_content)
                else:
                    # Handle Streamlit file objects or other buffer-like objects
                    if hasattr(file_content, 'read'):
                        content = file_content.read()
                        if isinstance(content, str):
                            f.write(content.encode('utf-8'))
                        else:
                            f.write(content)
                    else:
                        # Convert to bytes if possible
                        f.write(bytes(file_content))
            
            # Validate the saved file
            validation = self.validate_file(file_path, file_type)
            
            if not validation["valid"]:
                # Remove invalid file
                file_path.unlink(missing_ok=True)
                return {
                    "success": False,
                    "errors": validation["errors"],
                    "file_path": None,
                    "format": None  # Add format key
                }
            
            logger.info(f"File saved successfully: {file_path}")
            return {
                "success": True,
                "file_path": file_path,
                "original_filename": filename,
                "saved_filename": safe_filename,
                "file_size": validation["file_size"],
                "file_type": validation["file_type"],
                "format": validation["file_type"],  # Add format key for compatibility
                "mime_type": validation["mime_type"],
                "errors": []
            }
            
        except Exception as e:
            logger.error(f"Error saving file {filename}: {e}")
            return {
                "success": False,
                "errors": [f"Save error: {str(e)}"],
                "file_path": None,
                "format": None  # Add format key
            }
    
    # # ...existing code...
    # def save_file(file_path, content, binary=False):
    #     mode = 'wb' if binary else 'w'
    #     with open(file_path, mode) as f:
    #         if binary:
    #             if isinstance(content, str):
    #                 # If content is a string, encode it to bytes
    #                 f.write(content.encode('utf-8'))
    #             else:
    #                 f.write(content)
    #         else:
    #             if isinstance(content, bytes):
    #                 # If content is bytes, decode to string
    #                 f.write(content.decode('utf-8'))
    #             else:
    #                 f.write(content)
    # # ...existing code...

    def read_file(self, file_path: str) -> str:
        """
        Simple file reader for CLI mode.
        
        Args:
            file_path: Path to file to read
            
        Returns:
            File content as string
        """
        try:
            file_path = Path(file_path)
            
            # Handle different file types
            if file_path.suffix.lower() == '.pdf':
                # For PDF files, we'll need to extract text
                # For now, return a placeholder - PDF agent will handle extraction
                return f"PDF_FILE:{file_path}"
            else:
                # Handle text files
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except UnicodeDecodeError:
                    # Try with different encoding
                    with open(file_path, 'r', encoding='latin-1') as f:
                        return f.read()
                        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise Exception(f"Could not read file: {e}")

    def save_streamlit_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Save a Streamlit uploaded file.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
        
        Returns:
            Dict with save results
        """
        try:
            if uploaded_file is None:
                return {
                    "success": False,
                    "errors": ["No file provided"],
                    "file_path": None
                }
            
            # Read file content
            file_content = uploaded_file.read()
            
            # Reset file pointer for potential re-reading
            uploaded_file.seek(0)
            
            return self.save_uploaded_file(
                file_content=file_content,
                filename=uploaded_file.name,
                file_type=None  # Will auto-detect based on extension
            )
            
        except Exception as e:
            logger.error(f"Error saving Streamlit file: {e}")
            return {
                "success": False,
                "errors": [f"Streamlit file save error: {str(e)}"],
                "file_path": None
            }
    
    def read_file_content(self, file_path: Path, encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Read file content with appropriate handling for different file types.
        
        Args:
            file_path: Path to the file to read
            encoding: Text encoding for text files
        
        Returns:
            Dict with file content and metadata
        """
        try:
            if not file_path.exists():
                return {
                    "success": False,
                    "content": None,
                    "error": "File does not exist"
                }
            
            file_extension = file_path.suffix.lower()
            
            # Handle binary files (PDFs)
            if file_extension == '.pdf':
                with open(file_path, 'rb') as f:
                    content = f.read()
                return {
                    "success": True,
                    "content": content,
                    "content_type": "binary",
                    "encoding": None,
                    "size": len(content)
                }
            
            # Handle text files (JSON, email, etc.)
            else:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    return {
                        "success": True,
                        "content": content,
                        "content_type": "text",
                        "encoding": encoding,
                        "size": len(content.encode(encoding))
                    }
                except UnicodeDecodeError:
                    # Try with different encoding
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                    return {
                        "success": True,
                        "content": content,
                        "content_type": "text",
                        "encoding": "latin-1",
                        "size": len(content.encode('latin-1'))
                    }
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {
                "success": False,
                "content": None,
                "error": str(e)
            }
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old files from the upload directory.
        
        Args:
            max_age_hours: Maximum age of files to keep (in hours)
        
        Returns:
            Number of files deleted
        """
        try:
            current_time = datetime.now()
            deleted_count = 0
            
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age.total_seconds() > (max_age_hours * 3600):
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old file: {file_path}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during file cleanup: {e}")
            return 0
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get comprehensive information about a file.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Dict with file information
        """
        try:
            if not file_path.exists():
                return {"exists": False}
            
            stat = file_path.stat()
            mime_type, encoding = mimetypes.guess_type(str(file_path))
            
            return {
                "exists": True,
                "name": file_path.name,
                "size": stat.st_size,
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "extension": file_path.suffix.lower(),
                "mime_type": mime_type,
                "encoding": encoding,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "is_text": mime_type and mime_type.startswith('text/') if mime_type else False,
                "is_binary": not (mime_type and mime_type.startswith('text/')) if mime_type else True,
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {
                "exists": True,
                "error": str(e)
            }
    
    def delete_file(self, file_path: Path) -> bool:
        """
        Safely delete a file.
        
        Args:
            file_path: Path to the file to delete
        
        Returns:
            True if deletion was successful
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False


# Global file handler instance
file_handler = FileHandler()