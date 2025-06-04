"""
Base agent class providing common functionality for all agents.
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime

from memory.models import ProcessingRecord
from memory.database import db_manager
from utils.llm_client import llm_client

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, agent_name: str, 
                 llm_client: Optional[Any] = None, 
                 db_manager: Optional[Any] = None):
        self.agent_name = agent_name
        # Add agent_type for backward compatibility with tests
        self.agent_type = agent_name
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
        
        # Assign optional dependencies
        self.llm_client = llm_client or globals().get("llm_client")
        self.db_manager = db_manager or globals().get("db_manager")

    def handle_error(self, error: Exception, context: str = "Agent processing") -> Dict[str, Any]:
        """
        Handle errors and return standardized error response
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            
        Returns:
            Standardized error response dictionary
        """
        import traceback
        
        error_message = f"{context}: {str(error)}"
        
        # Log the error
        self.logger.error(error_message)
        self.logger.debug(f"Error traceback: {traceback.format_exc()}")
        
        # Store error in database if possible
        try:
            self.log_processing(
                source="error",
                format_type="ERROR",
                intent="error",
                extracted_fields={"error": error_message},
                confidence_score=0.0,
                status="ERROR",
                error_message=error_message
            )
        except Exception as db_error:
            self.logger.error(f"Failed to log error to database: {db_error}")
        
        return {
            "success": False,
            "error": error_message,
            "extracted_fields": {},
            "confidence_score": 0.0,
            "agent": self.agent_name,
            "status": "ERROR"
        }
    
    def log_processing(self, source: str, format_type: str, intent: str = "", 
                  extracted_fields: Optional[Dict] = None, 
                  confidence_score: float = 0.0, status: str = "pending",
                  error_message: str = "", thread_id: str = "", file_size: int = 0, processing_time: float = 0.0, **kwargs) -> Optional[int]:
        """
        Log processing information to the database.
        
        Args:
            source: Source filename or identifier
            format_type: File format type
            intent: Classified intent
            extracted_fields: Dictionary of extracted fields
            confidence_score: Confidence score (0.0-1.0)
            status: Processing status
            error_message: Error message if any
            thread_id: Thread ID for grouping
            file_size: Size of the file in bytes
            processing_time: Time taken to process in seconds
            **kwargs: Additional fields
        
        Returns:
            Record ID if successful, None otherwise
        """
        try:
            if not self.db_manager:
                self.logger.warning("No database manager available for logging")
                return None
            
            # Ensure source is a string, convert if bytes
            if isinstance(source, bytes):
                try:
                    source = source.decode('utf-8')
                except Exception:
                    source = str(source)
            elif not isinstance(source, str):
                source = str(source)
            
            record = ProcessingRecord(
                source=source,
                format=format_type,
                intent=intent,
                extracted_fields=extracted_fields or {},
                confidence_score=confidence_score,
                status=status,
                error_message=error_message,
                thread_id=thread_id,
                file_size=file_size,
                processing_time=processing_time,
                processed_by=self.agent_name
            )
            
            # Set any additional fields
            for key, value in kwargs.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            
            record_id = self.db_manager.create_record(record)
            self.logger.info(f"Logged processing record with ID: {record_id}")
            return record_id
            
        except Exception as e:
            self.logger.error(f"Failed to log processing: {e}")
            return None

    @abstractmethod
    def process(self, content: Union[str, Dict, Path], **kwargs) -> Dict[str, Any]:
        """
        Process content and return structured results.
        
        Args:
            content: Content to process (can be string, dict, or file path)
            **kwargs: Additional processing parameters
        
        Returns:
            Dictionary containing processing results
        """
        pass
    
    def _call_llm(self, prompt: str, **kwargs) -> str:
        """
        Call LLM with given prompt.
        
        Args:
            prompt: Prompt to send to LLM
            **kwargs: Additional parameters
        
        Returns:
            LLM response as string
        """
        try:
            if self.llm_client:
                response = self.llm_client.generate(prompt, **kwargs)
                return response.strip() if response else ""
            else:
                self.logger.warning("No LLM client available")
                return ""
                
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return ""
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """
        Create standardized error result.
        
        Args:
            error_message: Error message
        
        Returns:
            Error result dictionary
        """
        return {
            "agent": self.agent_type,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error_message": error_message,
            "confidence_score": 0.0,
            "extracted_fields": {},
            "success": False
        }
    
    def create_processing_record(self, source: str, file_format: str, 
                               intent: str = "", thread_id: str = "", 
                               **kwargs) -> ProcessingRecord:
        """
        Create a new processing record for tracking.
        
        Args:
            source: Source filename or identifier
            file_format: File format (pdf, json, email)
            intent: Classified intent
            thread_id: Thread ID for grouping related records
            **kwargs: Additional fields to set
        
        Returns:
            New ProcessingRecord instance
        """
        record = ProcessingRecord(
            source=source,
            format=file_format,
            intent=intent,
            thread_id=thread_id,
            status="pending"
        )
        
        # Set any additional fields
        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)
        
        return record
    
    def save_record(self, record: ProcessingRecord) -> int:
        """
        Save or update a processing record in the database.
        
        Args:
            record: Processing record to save
        
        Returns:
            Record ID
        """
        try:
            if record.id is None:
                # Create new record
                record_id = db_manager.create_record(record)
                record.id = record_id
                self.logger.info(f"Created new record with ID: {record_id}")
            else:
                # Update existing record
                db_manager.update_record(record)
                self.logger.info(f"Updated record {record.id}")
            
            return record.id
            
        except Exception as e:
            self.logger.error(f"Failed to save record: {e}")
            raise
    
    def update_record_status(self, record: ProcessingRecord, status: str, 
                           error_message: str = ""):
        """
        Update the status of a processing record.
        
        Args:
            record: Processing record to update
            status: New status
            error_message: Error message if status is 'failed'
        """
        record.update_status(status, error_message)
        if record.id:
            self.save_record(record)
    
    def measure_processing_time(self, func, *args, **kwargs):
        """
        Measure the execution time of a function.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Tuple of (result, processing_time)
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            processing_time = time.time() - start_time
            return result, processing_time
        except Exception as e:
            processing_time = time.time() - start_time
            raise e
    
    def safe_process(self, file_path: Path, record: ProcessingRecord) -> ProcessingRecord:
        """
        Safely process a file with error handling and timing.
        
        Args:
            file_path: Path to the file to process
            record: Processing record to update
        
        Returns:
            Updated processing record
        """
        self.logger.info(f"Starting processing with {self.agent_name}")
        
        try:
            # Update status to processing
            self.update_record_status(record, "processing")
            
            # Process with timing
            result, processing_time = self.measure_processing_time(
                self.process, file_path, record
            )
            
            # Update processing metadata
            result.set_processing_metadata(
                agent_name=self.agent_name,
                processing_time=processing_time,
                confidence=result.confidence_score
            )
            
            # Save the updated record
            self.save_record(result)
            
            self.logger.info(
                f"Successfully processed {file_path.name} in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Update record with error status
            record.update_status("failed", error_msg)
            record.processed_by = self.agent_name
            
            if record.id:
                self.save_record(record)
            
            return record
    
    def extract_with_llm(self, text: str, extraction_prompt: str) -> str:
        """
        Use LLM to extract information from text.
        
        Args:
            text: Text to process
            extraction_prompt: Prompt for extraction
        
        Returns:
            Extracted information as string
        """
        try:
            full_prompt = f"{extraction_prompt}\n\nText to process:\n{text}\n\nExtracted information:"
            response = self._call_llm(full_prompt)
            return response
            
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
            return ""
    
    def classify_with_llm(self, text: str, categories: List[str], *args, **kwargs) -> Dict[str, Any]:
        """
        Use LLM to classify text into categories.
        
        Args:
            text: Text to classify
            categories: List of possible categories
        
        Returns:
            Classification result dict
        """
        try:
            if self.llm_client and hasattr(self.llm_client, 'classify'):
                result = self.llm_client.classify(text, categories)
                return result
            else:
                # Fallback to generic LLM call
                prompt = f"Classify the following text into one of these categories: {', '.join(categories)}\n\nText: {text}\n\nCategory:"
                response = self._call_llm(prompt)
                return {
                    "category": response if response in categories else categories[0],
                    "confidence": 0.7,
                    "reasoning": f"LLM classification: {response}"
                }
            
        except Exception as e:
            self.logger.error(f"LLM classification failed: {e}")
            return {
                "category": categories[0] if categories else "Other",
                "confidence": 0.1,
                "reasoning": f"Classification failed: {str(e)}"
            }
    
    def validate_extracted_data(self, data: Dict[str, Any], 
                              required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate extracted data and calculate confidence score.
        
        Args:
            data: Extracted data dictionary
            required_fields: List of required field names
        
        Returns:
            Validation result with confidence score
        """
        if required_fields is None:
            required_fields = []
        
        validation_result = {
            "is_valid": True,
            "confidence_score": 1.0,
            "missing_fields": [],
            "empty_fields": [],
            "validation_errors": []
        }
        
        # Check for missing required fields
        for field in required_fields:
            if field not in data:
                validation_result["missing_fields"].append(field)
                validation_result["is_valid"] = False
        
        # Check for empty required fields
        for field in required_fields:
            if field in data and (data[field] is None or str(data[field]).strip() == ""):
                validation_result["empty_fields"].append(field)
                validation_result["is_valid"] = False
        
        # Calculate confidence score based on completeness
        total_fields = len(required_fields) if required_fields else len(data)
        if total_fields > 0:
            missing_count = len(validation_result["missing_fields"])
            empty_count = len(validation_result["empty_fields"])
            incomplete_count = missing_count + empty_count
            
            # Confidence decreases based on missing/empty fields
            validation_result["confidence_score"] = max(
                0.0, 1.0 - (incomplete_count / total_fields)
            )
        
        # Additional validation for data quality
        populated_fields = 0
        for key, value in data.items():
            if value is not None and str(value).strip():
                populated_fields += 1
        
        # Boost confidence if more fields are populated than required
        if len(data) > 0:
            population_ratio = populated_fields / len(data)
            validation_result["confidence_score"] = min(
                1.0, validation_result["confidence_score"] * (0.5 + 0.5 * population_ratio)
            )
        
        return validation_result
    
    def sanitize_text(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize and clean text for processing.
        
        Args:
            text: Text to sanitize
            max_length: Maximum length to truncate to
        
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Basic sanitization
        sanitized = str(text).strip()
        
        # Remove excessive whitespace
        sanitized = " ".join(sanitized.split())
        
        # Truncate if needed
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip() + "..."
        
        return sanitized
    
    def calculate_confidence_score(self, factors: Dict[str, float]) -> float:
        """
        Calculate overall confidence score from multiple factors.
        
        Args:
            factors: Dictionary of factor names and their scores (0.0-1.0)
        
        Returns:
            Overall confidence score (0.0-1.0)
        """
        if not factors:
            return 0.0
        
        # Simple weighted average (can be overridden by subclasses)
        total_score = sum(factors.values())
        return min(1.0, max(0.0, total_score / len(factors)))
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about this agent.
        
        Returns:
            Agent information dictionary
        """
        return {
            "name": self.agent_name,
            "class": self.__class__.__name__,
            "version": getattr(self, "version", "1.0.0"),
            "capabilities": getattr(self, "capabilities", [])
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.agent_name})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Create a concrete implementation for testing purposes
class TestableBaseAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    def process(self, content: Union[str, Dict, Path], **kwargs) -> Dict[str, Any]:
        """Test implementation of process method."""
        return {
            "agent": self.agent_type,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "content_type": type(content).__name__,
            "processed": True
        }