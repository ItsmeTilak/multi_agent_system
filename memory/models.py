"""
Database schema models for the multi-agent system.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import json


@dataclass
class ProcessingRecord:
    """
    Data model for tracking document processing across agents.
    """
    id: Optional[int] = None
    source: str = ""  # Original filename or source identifier
    format: str = ""  # File format (pdf, json, email)
    intent: str = ""  # Classified intent (RFQ, Invoice, etc.)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Email-specific fields
    sender: str = ""
    urgency: str = "normal"  # low, normal, high, critical
    
    # Processing metadata
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    processed_by: str = ""  # Agent that processed the document
    thread_id: str = ""  # For tracking grouped interactions
    
    # Quality and status tracking
    confidence_score: float = 0.0
    status: str = "pending"  # pending, processing, completed, failed
    error_message: str = ""
    
    # Additional metadata
    file_size: int = 0  # Size in bytes
    processing_time: float = 0.0  # Processing time in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a dictionary for database storage."""
        return {
            'id': self.id,
            'source': self.source,
            'format': self.format,
            'intent': self.intent,
            'timestamp': self.timestamp.isoformat(),
            'sender': self.sender,
            'urgency': self.urgency,
            'extracted_fields': json.dumps(self.extracted_fields),
            'processed_by': self.processed_by,
            'thread_id': self.thread_id,
            'confidence_score': self.confidence_score,
            'status': self.status,
            'error_message': self.error_message,
            'file_size': self.file_size,
            'processing_time': self.processing_time,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingRecord':
        """Create a ProcessingRecord from a dictionary."""
        # Parse timestamp
        timestamp = datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else datetime.now()
        
        # Parse extracted_fields JSON
        extracted_fields = {}
        if data.get('extracted_fields'):
            try:
                extracted_fields = json.loads(data['extracted_fields'])
            except json.JSONDecodeError:
                extracted_fields = {}
        
        return cls(
            id=data.get('id'),
            source=data.get('source', ''),
            format=data.get('format', ''),
            intent=data.get('intent', ''),
            timestamp=timestamp,
            sender=data.get('sender', ''),
            urgency=data.get('urgency', 'normal'),
            extracted_fields=extracted_fields,
            processed_by=data.get('processed_by', ''),
            thread_id=data.get('thread_id', ''),
            confidence_score=data.get('confidence_score', 0.0),
            status=data.get('status', 'pending'),
            error_message=data.get('error_message', ''),
            file_size=data.get('file_size', 0),
            processing_time=data.get('processing_time', 0.0),
        )
    
    def update_status(self, status: str, error_message: str = ""):
        """Update the processing status."""
        self.status = status
        if error_message:
            self.error_message = error_message
    
    def add_extracted_field(self, key: str, value: Any):
        """Add an extracted field to the record."""
        self.extracted_fields[key] = value
    
    def set_processing_metadata(self, agent_name: str, processing_time: float, confidence: float = 0.0):
        """Set processing metadata after agent completion."""
        self.processed_by = agent_name
        self.processing_time = processing_time
        self.confidence_score = confidence
        self.status = "completed"


# Database schema SQL statements
CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS processing_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    format TEXT NOT NULL,
    intent TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    sender TEXT DEFAULT '',
    urgency TEXT DEFAULT 'normal',
    extracted_fields TEXT DEFAULT '{}',
    processed_by TEXT DEFAULT '',
    thread_id TEXT DEFAULT '',
    confidence_score REAL DEFAULT 0.0,
    status TEXT DEFAULT 'pending',
    error_message TEXT DEFAULT '',
    file_size INTEGER DEFAULT 0,
    processing_time REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_format ON processing_records(format);
CREATE INDEX IF NOT EXISTS idx_intent ON processing_records(intent);
CREATE INDEX IF NOT EXISTS idx_status ON processing_records(status);
CREATE INDEX IF NOT EXISTS idx_thread_id ON processing_records(thread_id);
CREATE INDEX IF NOT EXISTS idx_timestamp ON processing_records(timestamp);

-- Trigger to update the updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_processing_records_timestamp 
    AFTER UPDATE ON processing_records
    BEGIN
        UPDATE processing_records SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;
"""