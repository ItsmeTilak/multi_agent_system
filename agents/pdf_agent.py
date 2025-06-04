"""
PDF Agent for parsing and extracting data from PDF documents.
Uses pdfplumber for text extraction and LLM for intelligent field extraction.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pdfplumber
import io
from pathlib import Path

from .base_agent import BaseAgent
from utils.llm_client import BaseLLMClient, llm_client
from typing import Optional
from memory.database import DatabaseManager

logger = logging.getLogger(__name__)


class PDFAgent(BaseAgent):
    """
    Specialized agent for processing PDF documents.
    Extracts text, tables, and structured data using OCR and NLP techniques.
    """

    def __init__(self, db_manager=None, llm_client=None):
        super().__init__("pdf_agent", llm_client, db_manager)
        self.extraction_patterns = self._init_extraction_patterns()
        
    def _init_extraction_patterns(self) -> Dict[str, Dict[str, str]]:
        """Initialize regex patterns for common PDF field extraction."""
        return {
            "invoice": {
                "invoice_number": r"(?:invoice|inv)\s*(?:number|#|no)[\s:]*([A-Z0-9-]+)",
                "date": r"(?:date|issued)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
                "amount": r"(?:total|amount|due)[\s:]*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)",
                "vendor": r"(?:from|vendor|company)[\s:]*([A-Za-z\s&.,]+?)(?:\n|$)",
                "customer": r"(?:to|bill\s+to|customer)[\s:]*([A-Za-z\s&.,]+?)(?:\n|$)"
            },
            "complaint": {
                "case_number": r"(?:case|complaint|ticket)\s*(?:number|#|no)[\s:]*([A-Z0-9-]+)",
                "date": r"(?:date|filed|submitted)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
                "complainant": r"(?:complainant|filed\s+by|name)[\s:]*([A-Za-z\s.,]+?)(?:\n|$)",
                "subject": r"(?:subject|regarding|issue)[\s:]*([^\n]+)",
                "priority": r"(?:priority|urgency|level)[\s:]*([A-Za-z]+)"
            },
            "rfq": {
                "rfq_number": r"(?:rfq|request)\s*(?:number|#|no)[\s:]*([A-Z0-9-]+)",
                "due_date": r"(?:due|deadline|by)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
                "project": r"(?:project|for)[\s:]*([^\n]+)",
                "contact": r"(?:contact|email|phone)[\s:]*([A-Za-z0-9@._\s-]+)"
            },
            "report": {
                "title": r"(?:title|report|subject)[\s:]*([^\n]+)",
                "date": r"(?:date|prepared|generated)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
                "author": r"(?:author|prepared\s+by|by)[\s:]*([A-Za-z\s.,]+?)(?:\n|$)",
                "department": r"(?:department|division|unit)[\s:]*([A-Za-z\s&.,]+?)(?:\n|$)"
            }
        }
    
    def process(self, file_path: str, classification_result: Optional[Dict] = None, source_name: Optional[str] = None, file_size: Optional[int] = None, processing_time: Optional[float] = None, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process PDF file and extract structured data.
        
        Args:
            file_path: Path to the PDF file
            classification_result: Classification result from classifier agent
            source_name: Optional source name for logging and database
            file_size: Optional file size in bytes
            processing_time: Optional processing time in seconds
            thread_id: Optional thread ID for grouping
            
        Returns:
            Dict containing processing results and extracted data
        """
        try:
            source_str = file_path
            if isinstance(file_path, bytes):
                source_str = "uploaded_pdf_content"
            else:
                source_str = str(file_path)
            logger.info(f"Processing PDF file: {source_str}")
            logger.info(f"Processing PDF content with {self.agent_type}")
            logger.info(f"Classification result: {classification_result}")
            
            # Initialize result structure
            result = {
                "agent": self.agent_type,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "confidence_score": 0.0,
                "extracted_fields": {},
                "document_info": {},
                "text_analysis": {},
                "tables": [],
                "recommendations": []
            }
            # Extract text and metadata from PDF
            pdf_data = self._extract_pdf_content(file_path)
            result["document_info"] = pdf_data["metadata"]
            result["text_analysis"] = pdf_data["text_analysis"]
            result["tables"] = pdf_data["tables"]
            
            # Classify document type
            doc_type = self._classify_document(pdf_data["text"])
            result["document_type"] = doc_type
            
            # Extract structured fields
            extracted_fields = self._extract_structured_fields(pdf_data["text"], doc_type)
            result["extracted_fields"] = extracted_fields
            
            # Perform advanced analysis with LLM if available
            if self.llm_client and pdf_data["text"]:
                llm_analysis = self._llm_extract_fields(pdf_data["text"], doc_type)
                result["llm_analysis"] = llm_analysis
                # Merge LLM extracted fields
                if "extracted_fields" in llm_analysis:
                    result["extracted_fields"].update(llm_analysis["extracted_fields"])
            
            # Generate recommendations
            result["recommendations"] = self._generate_recommendations(result)
            
            # Calculate confidence score
            result["confidence_score"] = self._calculate_confidence(result)
            
            source_str = source_name if source_name else None
            if not source_str:
                if isinstance(file_path, bytes):
                    source_str = "uploaded_pdf_content"
                else:
                    source_str = str(file_path)
            self.log_processing(
                source=source_str,
                format_type="PDF",
                intent=classification_result.get("intent", "unknown") if classification_result else "unknown",
                extracted_fields=result["extracted_fields"],
                confidence_score=result["confidence_score"],
                status="completed",
                thread_id=thread_id,
                file_size=file_size,
                processing_time=processing_time
            )

            logger.info(f"PDF processing completed with confidence: {result['confidence_score']}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            error_result = self.handle_error(e, "PDF processing error")
            
            # Log the error to database
            self.log_processing(
                source=source_name or file_path,
                format_type="PDF", 
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

    def _extract_pdf_content(self, content: Union[str, bytes]) -> Dict[str, Any]:
        """
        Extract text, tables, and metadata from PDF.
        
        Args:
            content: PDF file path or binary content
            
        Returns:
            Dict containing extracted PDF data
        """
        pdf_data = {
            "text": "",
            "tables": [],
            "metadata": {},
            "text_analysis": {}
        }
        
        try:
            # Handle different input types
            if isinstance(content, str):
                # Check if it's a FileHandler-formatted string
                if content.startswith("PDF_FILE:"):
                    # Extract the actual file path
                    actual_file_path = content.replace("PDF_FILE:", "")
                    if Path(actual_file_path).exists():
                        with pdfplumber.open(actual_file_path) as pdf:
                            pdf_data = self._process_pdf_object(pdf)
                    else:
                        raise ValueError(f"PDF file not found: {actual_file_path}")
                elif Path(content).exists():
                    # Direct file path
                    with pdfplumber.open(content) as pdf:
                        pdf_data = self._process_pdf_object(pdf)
                else:
                    raise ValueError("Invalid PDF file path")
            elif isinstance(content, bytes):
                # Binary content
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    pdf_data = self._process_pdf_object(pdf)
            else:
                raise ValueError("Invalid PDF content format")
                
        except Exception as e:
            logger.error(f"Error extracting PDF content: {str(e)}")
            pdf_data["error"] = str(e)
        
        return pdf_data
    
    def _process_pdf_object(self, pdf) -> Dict[str, Any]:
        """Process pdfplumber PDF object to extract content."""
        all_text = []
        all_tables = []
        
        # Extract metadata
        metadata = {
            "page_count": len(pdf.pages),
            "creator": pdf.metadata.get('Creator', ''),
            "producer": pdf.metadata.get('Producer', ''),
            "creation_date": str(pdf.metadata.get('CreationDate', '')),
            "title": pdf.metadata.get('Title', ''),
            "subject": pdf.metadata.get('Subject', '')
        }
        
        # Process each page
        for page_num, page in enumerate(pdf.pages):
            try:
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    all_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
                
                # Extract tables
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if table:
                        all_tables.append({
                            "page": page_num + 1,
                            "table_index": table_idx,
                            "data": table,
                            "rows": len(table),
                            "columns": len(table[0]) if table else 0
                        })
                        
            except Exception as e:
                logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
        
        # Combine all text
        full_text = "\n".join(all_text)
        
        # Analyze text characteristics
        text_analysis = self._analyze_text_structure(full_text)
        
        return {
            "text": full_text,
            "tables": all_tables,
            "metadata": metadata,
            "text_analysis": text_analysis
        }
    
    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure and characteristics."""
        if not text:
            return {}
        
        # Basic text statistics
        lines = text.split('\n')
        words = text.split()
        
        # Detect common patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}'
        currency_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
        
        emails = re.findall(email_pattern, text, re.IGNORECASE)
        phones = re.findall(phone_pattern, text)
        dates = re.findall(date_pattern, text)
        currencies = re.findall(currency_pattern, text)
        
        return {
            "total_characters": len(text),
            "total_words": len(words),
            "total_lines": len(lines),
            "average_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
            "emails_found": len(emails),
            "phones_found": len(phones),
            "dates_found": len(dates),
            "currencies_found": len(currencies),
            "detected_emails": emails[:5],  # First 5 emails
            "detected_phones": phones[:5],  # First 5 phone numbers
            "detected_dates": dates[:5],    # First 5 dates
            "detected_currencies": currencies[:5]  # First 5 currency amounts
        }
    
    def _classify_document(self, text: str) -> str:
        """
        Classify the document type based on content patterns.
        
        Args:
            text: Extracted PDF text
            
        Returns:
            Classified document type
        """
        if not text:
            return "unknown"
        
        text_lower = text.lower()
        
        # Invoice patterns
        invoice_keywords = ["invoice", "bill", "payment", "amount due", "total", "tax"]
        invoice_score = sum(1 for keyword in invoice_keywords if keyword in text_lower)
        
        # Complaint patterns
        complaint_keywords = ["complaint", "issue", "problem", "concern", "dissatisfied", "resolve"]
        complaint_score = sum(1 for keyword in complaint_keywords if keyword in text_lower)
        
        # RFQ patterns
        rfq_keywords = ["request for quote", "rfq", "proposal", "bid", "quotation", "tender"]
        rfq_score = sum(1 for keyword in rfq_keywords if keyword in text_lower)
        
        # Report patterns
        report_keywords = ["report", "analysis", "summary", "findings", "conclusion", "executive"]
        report_score = sum(1 for keyword in report_keywords if keyword in text_lower)
        
        # Contract patterns
        contract_keywords = ["agreement", "contract", "terms", "conditions", "party", "whereas"]
        contract_score = sum(1 for keyword in contract_keywords if keyword in text_lower)
        
        # Determine document type based on highest score
        scores = {
            "invoice": invoice_score,
            "complaint": complaint_score,
            "rfq": rfq_score,
            "report": report_score,
            "contract": contract_score
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            return "general"
        
        return max(scores, key=scores.get)
    
    def _extract_structured_fields(self, text: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract structured fields using regex patterns.
        
        Args:
            text: PDF text content
            doc_type: Classified document type
            
        Returns:
            Dictionary of extracted fields
        """
        extracted = {}
        
        if not text or doc_type not in self.extraction_patterns:
            return extracted
        
        patterns = self.extraction_patterns[doc_type]
        
        for field_name, pattern in patterns.items():
            try:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    # Take the first match and clean it
                    value = matches[0].strip()
                    if value:
                        extracted[field_name] = value
            except Exception as e:
                logger.warning(f"Error extracting {field_name}: {str(e)}")
        
        # Additional extraction based on document type
        if doc_type == "invoice":
            extracted.update(self._extract_invoice_specifics(text))
        elif doc_type == "complaint":
            extracted.update(self._extract_complaint_specifics(text))
        elif doc_type == "rfq":
            extracted.update(self._extract_rfq_specifics(text))
        
        # Extract common fields
        extracted.update(self._extract_common_fields(text))
        
        return extracted
    
    def _extract_invoice_specifics(self, text: str) -> Dict[str, Any]:
        """Extract invoice-specific fields."""
        specifics = {}
        
        # Extract line items (simplified)
        line_items = []
        lines = text.split('\n')
        for line in lines:
            if re.search(r'\$\d+', line) and any(word in line.lower() for word in ['qty', 'quantity', 'item', 'description']):
                line_items.append(line.strip())
        
        if line_items:
            specifics["line_items"] = line_items[:10]  # First 10 items
        
        # Extract tax information
        tax_match = re.search(r'tax[\s:]*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', text, re.IGNORECASE)
        if tax_match:
            specifics["tax_amount"] = tax_match.group(1)
        
        return specifics
    
    def _extract_complaint_specifics(self, text: str) -> Dict[str, Any]:
        """Extract complaint-specific fields."""
        specifics = {}
        
        # Extract complaint description (first paragraph after keywords)
        desc_pattern = r'(?:description|details|issue)[\s:]*([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|$)'
        desc_match = re.search(desc_pattern, text, re.IGNORECASE | re.MULTILINE)
        if desc_match:
            specifics["description"] = desc_match.group(1).strip()[:500]  # Limit length
        
        # Extract resolution status
        status_keywords = ["resolved", "pending", "open", "closed", "in progress"]
        for keyword in status_keywords:
            if keyword in text.lower():
                specifics["status"] = keyword
                break
        
        return specifics
    
    def _extract_rfq_specifics(self, text: str) -> Dict[str, Any]:
        """Extract RFQ-specific fields."""
        specifics = {}
        
        # Extract requirements
        req_pattern = r'(?:requirements|specifications|scope)[\s:]*([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|$)'
        req_match = re.search(req_pattern, text, re.IGNORECASE | re.MULTILINE)
        if req_match:
            specifics["requirements"] = req_match.group(1).strip()[:500]
        
        # Extract delivery terms
        delivery_pattern = r'(?:delivery|timeline|schedule)[\s:]*([^\n]+)'
        delivery_match = re.search(delivery_pattern, text, re.IGNORECASE)
        if delivery_match:
            specifics["delivery_terms"] = delivery_match.group(1).strip()
        
        return specifics
    
    def _extract_common_fields(self, text: str) -> Dict[str, Any]:
        """Extract common fields present in most documents."""
        common = {}
        
        # Extract all email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            common["emails"] = list(set(emails))[:5]  # Unique emails, max 5
        
        # Extract all phone numbers
        phones = re.findall(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        if phones:
            common["phone_numbers"] = list(set(phones))[:5]
        
        # Extract addresses (simplified)
        address_pattern = r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)[^\n]*'
        addresses = re.findall(address_pattern, text)
        if addresses:
            common["addresses"] = addresses[:3]
        
        # Extract company names (capitalized words near common business terms)
        company_pattern = r'(?:company|corp|corporation|inc|llc|ltd)[\s:]*([A-Z][A-Za-z\s&.,]+?)(?:\n|$)'
        companies = re.findall(company_pattern, text, re.IGNORECASE)
        if companies:
            common["companies"] = [c.strip() for c in companies][:3]
        
        return common
    
    def _llm_extract_fields(self, text: str, doc_type: str) -> Dict[str, Any]:
        """
        Use LLM for advanced field extraction and analysis.
        
        Args:
            text: PDF text content
            doc_type: Document type
            
        Returns:
            LLM analysis results
        """
        try:
            # Truncate text for LLM processing
            truncated_text = text[:2000] if len(text) > 2000 else text
            
            prompt = f"""
            Analyze this {doc_type} document and extract key information:
            
            Document Text:
            {truncated_text}
            
            Please extract the following information in JSON format:
            1. Key entities (names, organizations, locations)
            2. Important dates and numbers
            3. Document purpose and main points
            4. Action items or next steps (if any)
            5. Data quality assessment (1-10 scale)
            
            Respond with a JSON object containing:
            - extracted_fields: {{key-value pairs of important data}}
            - entities: {{names, organizations, locations}}
            - summary: {{brief document summary}}
            - quality_score: {{data quality rating 1-10}}
            - insights: {{key insights or recommendations}}
            """
            
            response = self.llm_client.generate(prompt, max_tokens=800)
            
            try:
                import json
                llm_result = json.loads(response)
                return llm_result
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw response
                return {"raw_response": response}
                
        except Exception as e:
            logger.warning(f"LLM field extraction failed: {str(e)}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on processing results."""
        recommendations = []
        
        # Document quality recommendations
        if result.get("document_info", {}).get("page_count", 0) > 20:
            recommendations.append("Consider document summarization for large PDFs")
        
        # Text quality recommendations
        text_analysis = result.get("text_analysis", {})
        if text_analysis.get("total_words", 0) < 50:
            recommendations.append("Document appears to have minimal text content - check for scan quality")
        
        # Extraction quality recommendations
        extracted_fields = result.get("extracted_fields", {})
        if len(extracted_fields) < 3:
            recommendations.append("Limited structured data extracted - consider manual review")
        
        # Data completeness recommendations
        doc_type = result.get("document_type", "unknown")
        if doc_type == "invoice" and "amount" not in extracted_fields:
            recommendations.append("Invoice amount not detected - verify document completeness")
        
        if doc_type == "complaint" and "description" not in extracted_fields:
            recommendations.append("Complaint description not found - check document structure")
        
        # Table recommendations
        if result.get("tables"):
            recommendations.append("Document contains tables - consider structured data extraction")
        
        if not recommendations:
            recommendations.append("Document processing completed successfully")
        
        return recommendations
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score based on extraction results."""
        base_score = 0.5  # Base confidence
        
        # Boost for successful text extraction
        text_analysis = result.get("text_analysis", {})
        if text_analysis.get("total_words", 0) > 100:
            base_score += 0.2
        
        # Boost for structured field extraction
        extracted_fields = result.get("extracted_fields", {})
        field_count = len(extracted_fields)
        base_score += min(0.2, field_count * 0.05)
        
        # Boost for document type classification
        if result.get("document_type") != "unknown":
            base_score += 0.1
        
        # Boost for LLM analysis
        if result.get("llm_analysis") and "error" not in result["llm_analysis"]:
            base_score += 0.1
        
        # Boost for table extraction
        if result.get("tables"):
            base_score += 0.1
        
        return min(1.0, base_score)
