"""
JSON Agent for validating and extracting data from JSON inputs.
Handles schema validation, field extraction, and anomaly detection.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import jsonschema
from jsonschema import validate, ValidationError
import re
from utils.llm_client import BaseLLMClient, llm_client
from memory.database import DatabaseManager
from typing import Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class JSONAgent(BaseAgent):
    """
    Specialized agent for processing JSON data.
    Validates schema, extracts relevant fields, and detects anomalies.
    """

    def __init__(self, db_manager=None, llm_client=None):
        super().__init__("json_agent", llm_client, db_manager)
        self.common_schemas = self._load_common_schemas()
        
    def _load_common_schemas(self) -> Dict[str, Dict]:
        """Load common JSON schemas for validation."""
        return {
            "invoice": {
                "type": "object",
                "required": ["header", "summary"],
                "properties": {
                    "header": {
                        "type": "object",
                        "required": ["invoice_number", "invoice_date", "due_date"],
                        "properties": {
                            "invoice_number": {"type": "string"},
                            "invoice_date": {"type": "string"},
                            "due_date": {"type": "string"}
                        }
                    },
                    "summary": {
                        "type": "object",
                        "required": ["total_amount"],
                        "properties": {
                            "total_amount": {"type": "number"}
                        }
                    },
                    "vendor": {"type": "object"},
                    "customer": {"type": "object"},
                    "line_items": {"type": "array"}
                }
            },
            "order": {
                "type": "object",
                "required": ["order_id", "customer", "items"],
                "properties": {
                    "order_id": {"type": "string"},
                    "customer": {"type": "string"},
                    "items": {"type": "array"},
                    "total": {"type": "number"},
                    "status": {"type": "string"}
                }
            },
            "product": {
                "type": "object",
                "required": ["id", "name", "price"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "price": {"type": "number"},
                    "category": {"type": "string"},
                    "description": {"type": "string"}
                }
            }
        }
    
    def _detect_schema_type(self, json_data: Any) -> str:
        """Detect the schema type of the JSON data based on keys."""
        if not isinstance(json_data, dict):
            return "unknown"
        
        # Unwrap top-level key if single key and nested dict
        if len(json_data) == 1:
            first_key = next(iter(json_data))
            nested = json_data[first_key]
            if isinstance(nested, dict):
                keys = set(nested.keys())
                for schema_name, schema in self.common_schemas.items():
                    required_keys = set(schema.get("required", []))
                    if required_keys.issubset(keys):
                        return schema_name
        
        # Otherwise check top-level keys
        keys = set(json_data.keys())
        for schema_name, schema in self.common_schemas.items():
            required_keys = set(schema.get("required", []))
            if required_keys.issubset(keys):
                return schema_name
        return "unknown"
    
    def _validate_json_schema(self, json_data: Any, schema_type: str) -> Dict[str, Any]:
        """Validate JSON data against the detected schema."""
        schema = self.common_schemas.get(schema_type)
        if not schema:
            logger.debug(f"Unknown schema type: {schema_type}")
            return {"valid": False, "errors": ["Unknown schema type"]}
        
        try:
            # If nested, validate nested dict
            if len(json_data) == 1:
                first_key = next(iter(json_data))
                nested = json_data[first_key]
                if isinstance(nested, dict):
                    validate(instance=nested, schema=schema)
                    logger.debug(f"Validation succeeded for nested JSON with schema {schema_type}")
                    return {"valid": True, "errors": []}
            # Otherwise validate top-level
            validate(instance=json_data, schema=schema)
            logger.debug(f"Validation succeeded for top-level JSON with schema {schema_type}")
            return {"valid": True, "errors": []}
        except ValidationError as e:
            logger.debug(f"Validation failed: {str(e)}")
            return {"valid": False, "errors": [str(e)]}
    
    def _extract_fields(self, json_data: Any, schema_type: str) -> Dict[str, Any]:
        """Extract relevant fields based on schema type."""
        if not isinstance(json_data, dict):
            logger.debug("Input json_data is not a dict")
            return {}
        
        # Unwrap top-level key if single key and nested dict
        if len(json_data) == 1:
            first_key = next(iter(json_data))
            nested = json_data[first_key]
            if isinstance(nested, dict):
                json_data = nested
        
        extracted = {}
        if schema_type == "invoice":
            header = json_data.get("header", {})
            extracted["invoice_number"] = header.get("invoice_number")
            extracted["invoice_date"] = header.get("invoice_date")
            extracted["due_date"] = header.get("due_date")
            extracted["amount"] = json_data.get("summary", {}).get("total_amount")
            
            vendor = json_data.get("vendor", {})
            extracted["vendor_company_name"] = vendor.get("company_name")
            vendor_contact = vendor.get("contact", {})
            extracted["vendor_phone"] = vendor_contact.get("phone")
            extracted["vendor_email"] = vendor_contact.get("email")
            
            customer = json_data.get("customer", {})
            extracted["customer_company_name"] = customer.get("company_name")
            extracted["customer_contact_person"] = customer.get("contact_person")
            customer_contact = customer.get("contact", {})
            extracted["customer_phone"] = customer_contact.get("phone")
            extracted["customer_email"] = customer_contact.get("email")
            
            line_items = json_data.get("line_items", [])
            simplified_items = []
            for item in line_items:
                simplified_items.append({
                    "item_id": item.get("item_id"),
                    "description": item.get("description"),
                    "quantity": item.get("quantity"),
                    "unit_price": item.get("unit_price"),
                    "total_price": item.get("total_price")
                })
            extracted["line_items"] = simplified_items
        elif schema_type == "order":
            extracted["order_id"] = json_data.get("order_id")
            extracted["customer"] = json_data.get("customer")
            extracted["items"] = json_data.get("items")
            extracted["total"] = json_data.get("total")
            extracted["status"] = json_data.get("status")
        elif schema_type == "product":
            extracted["id"] = json_data.get("id")
            extracted["name"] = json_data.get("name")
            extracted["price"] = json_data.get("price")
            extracted["category"] = json_data.get("category")
            extracted["description"] = json_data.get("description")
        else:
            # For unknown schema, return empty or raw data summary
            # Remove raw_data key to avoid displaying raw data on UI
            # Instead, return empty dict or minimal summary
            return {}
        
        return extracted
    
    def _detect_anomalies(self, json_data: Any, extracted_fields: Dict[str, Any]) -> List[str]:
        """Detect anomalies in JSON data."""
        anomalies = []
        # Example: check for missing required fields
        if not extracted_fields:
            anomalies.append("No fields extracted")
        # Additional anomaly detection logic can be added here
        return anomalies
    
    def _generate_recommendations(self, json_data: Any, validation_result: Dict[str, Any], anomalies: List[str]) -> List[str]:
        """Generate recommendations based on validation and anomalies."""
        recommendations = []
        if not validation_result.get("valid", False):
            recommendations.append("JSON data failed schema validation")
        if anomalies:
            recommendations.append("Anomalies detected in JSON data")
        if not recommendations:
            recommendations.append("JSON processing completed successfully")
        return recommendations
    
    def _calculate_confidence(self, validation_result: Dict[str, Any], anomalies: List[str]) -> float:
        """Calculate confidence score based on validation and anomalies."""
        if not validation_result.get("valid", False):
            return 0.0
        if anomalies:
            return 0.5
        return 1.0
    
    def _llm_analyze(self, json_data: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced analysis using LLM."""
        # Placeholder for LLM analysis implementation
        return {}
    
    def process(self, content: Union[str, bytes], classification_result: Optional[Dict] = None, source_name: Optional[str] = None, file_size: Optional[int] = None, processing_time: Optional[float] = None, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process JSON content and extract structured data.
        
        Args:
            content: JSON content as string or bytes
            classification_result: Optional classification result
            source_name: Optional source name for logging and database
            file_size: Optional file size in bytes
            processing_time: Optional processing time in seconds
            thread_id: Optional thread ID for grouping

        Returns:
            Dict containing processing results and extracted data
        """
        try:
            log_source = source_name if source_name else "unknown_source"
            logger.info(f"Processing JSON content from source: {log_source}")
            logger.info(f"Processing JSON content with {self.agent_type}")
            
            # Parse JSON content
            if isinstance(content, bytes):
                content_str = content.decode('utf-8')
            else:
                content_str = content
            
            try:
                json_data = json.loads(content_str)
            except json.JSONDecodeError as e:
                return self._create_error_result(f"Invalid JSON format: {str(e)}")
            
            # Extract basic metadata
            result = {
                "agent": self.agent_type,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "confidence_score": 0.0,
                "extracted_fields": {},
                "validation_results": {},
                "anomalies": [],
                "recommendations": []
            }
            
            # Detect JSON schema type
            schema_type = self._detect_schema_type(json_data)
            result["detected_schema"] = schema_type
            
            # Validate against schema
            validation_result = self._validate_json_schema(json_data, schema_type)
            result["validation_results"] = validation_result
            
            # Extract structured fields
            extracted_fields = self._extract_fields(json_data, schema_type)
            result["extracted_fields"] = extracted_fields
            
            # Detect anomalies
            anomalies = self._detect_anomalies(json_data, extracted_fields)
            result["anomalies"] = anomalies
            
            # Generate recommendations
            recommendations = self._generate_recommendations(json_data, validation_result, anomalies)
            result["recommendations"] = recommendations
            
            # Calculate confidence score
            result["confidence_score"] = self._calculate_confidence(validation_result, anomalies)
            
            # Use LLM for advanced analysis if available
            if self.llm_client:
                llm_analysis = self._llm_analyze(json_data, result)
                result["llm_analysis"] = llm_analysis

            # Log processing results to database
            self.log_processing(
                source=log_source,
                format_type="JSON",
                intent=classification_result.get("intent", "unknown") if classification_result else "unknown",
                extracted_fields=extracted_fields,
                confidence_score=result["confidence_score"],
                status="completed",
                thread_id=thread_id,
                file_size=file_size,
                processing_time=processing_time
            )

            logger.info(f"JSON processing completed with confidence: {result['confidence_score']}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing JSON: {str(e)}")
            error_result = self.handle_error(e, "JSON processing error")
            
            # Log the error to database
            self.log_processing(
                source=log_source,
                format_type="JSON", 
                intent="unknown",
                extracted_fields={},
                confidence_score=0.0,
                status="error"
            )
            
            return error_result
    
    def handle_error(self, error: Exception, message: str = "Processing failed") -> Dict[str, Any]:
        """Handle errors and return error result"""
        error_msg = f"{message}: {str(error)}"
        logger.error(error_msg)
        
        return {
            "success": False,
            "error": error_msg,
            "agent": self.agent_type,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "confidence_score": 0.0,
            "extracted_fields": {},
            "recommendations": ["Manual review required due to processing error"]
        }
