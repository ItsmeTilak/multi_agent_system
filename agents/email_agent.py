"""
Email Agent
Processes email content and extracts structured information
"""

import re
import email
import base64
import os
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .base_agent import BaseAgent
from agents.pdf_agent import PDFAgent
from utils.file_handler import FileHandler

class EmailAgent(BaseAgent):
    """Agent for processing email content and attachments"""
    
    def __init__(self, db_manager=None, llm_client=None):
        super().__init__("email", llm_client, db_manager)
        
        self.file_handler = FileHandler()
        self.pdf_agent = PDFAgent(db_manager, llm_client)
        
        # Urgency keywords and patterns
        self.urgency_patterns = {
            "high": ["urgent", "asap", "immediately", "critical", "emergency", "rush"],
            "medium": ["soon", "priority", "important", "expedite"],
            "low": ["when convenient", "no rush", "whenever possible"]
        }
        
        # Common email patterns
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            "money": r'\$\s?\d+(?:,\d{3})*(?:\.\d{2})?',
            "order_number": r'(?:order|invoice|ref|reference)(?:\s*#?\s*:?\s*)([A-Z0-9-]+)',
        }
    
    def process(self, input_data: Any, context: Optional[Dict] = None, source_name: Optional[str] = None, file_size: Optional[int] = None, processing_time: Optional[float] = None, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process email content and extract structured information, including PDF attachments
        
        Args:
            input_data: Email content as string
            context: Optional context from classifier
            source_name: Optional source name for logging and database
            file_size: Optional file size in bytes
            processing_time: Optional processing time in seconds
            thread_id: Optional thread ID for grouping
            
        Returns:
            Structured email data and PDF attachment processing results
        """
        try:
            if not self.validate_extracted_data(input_data):
                return self.handle_error(ValueError("Invalid email input"))
            
            content = str(input_data)
            
            # Parse email message from content
            msg = email.message_from_string(content)
            
            # Extract basic email fields from headers and body
            email_data = self._extract_basic_fields(content)
            
            # Extract entities using patterns
            entities = self._extract_entities(content)
            email_data.update(entities)
            
            # Determine urgency
            email_data["urgency"] = self._determine_urgency(content)
            
            # Use LLM for advanced extraction
            llm_data = self._llm_extraction(content)
            
            if isinstance(llm_data, dict):
                email_data.update(llm_data)
            else:
                self.logger.warning(f"LLM data is not a dictionary: {type(llm_data)}, value: {llm_data}")
                if isinstance(llm_data, str):
                    email_data["llm_response"] = llm_data
                else:
                    email_data["llm_data"] = str(llm_data)
            
            # Process PDF attachments if any
            pdf_results = []
            for part in msg.walk():
                content_disposition = part.get("Content-Disposition", "")
                if part.get_content_maintype() == "application" and "pdf" in part.get_content_type():
                    if "attachment" in content_disposition:
                        filename = part.get_filename()
                        if filename:
                            # Save attachment to uploads directory
                            payload = part.get_payload(decode=True)
                            save_result = self.file_handler.save_uploaded_file(payload, filename, file_type="pdf")
                            if save_result["success"]:
                                # Rename the attachment file to include source .eml file reference
                                if source_name:
                                    eml_base = Path(source_name).stem
                                    ext = Path(filename).suffix
                                    new_filename = f"{eml_base}_{Path(filename).stem}{ext}"
                                    new_file_path = save_result["file_path"].parent / new_filename
                                    os.rename(save_result["file_path"], new_file_path)
                                    pdf_file_path = new_file_path
                                else:
                                    pdf_file_path = save_result["file_path"]
                                
                                # Generate file size for the attachment
                                attachment_file_size = os.path.getsize(pdf_file_path)
                                
                                # Pass thread_id and file_size to PDF agent process method
                                pdf_result = self.pdf_agent.process(
                                    str(pdf_file_path),
                                    context,
                                    source_name=new_filename if source_name else filename,
                                    file_size=attachment_file_size,
                                    thread_id=thread_id
                                )
                                pdf_results.append({
                                    "filename": new_filename if source_name else filename,
                                    "result": pdf_result
                                })
            
            # Calculate confidence for email content
            confidence = self.calculate_confidence(email_data)
            
            result = {
                "success": True,
                "extracted_fields": email_data,
                "confidence_score": confidence,
                "pdf_attachments": pdf_results,
                "agent": self.agent_name
            }
            
            if isinstance(context, dict):
                intent = context.get("intent", "unknown")
                thread_id_val = context.get("thread_id")
            else:
                intent = "unknown"
                thread_id_val = None
                
            log_source = source_name if source_name else "email_content"
            self.log_processing(
                source=log_source,
                format_type="EMAIL",
                intent=intent,
                extracted_fields=email_data,
                confidence_score=confidence,
                status="completed",
                thread_id=thread_id or thread_id_val,
                file_size=file_size,
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            return self.handle_error(e, "Email processing failed")

    def calculate_confidence(self, email_data: Dict[str, Any]) -> float:
        """Calculate confidence score for email extraction"""
        try:
            factors = {}
            
            # Check if basic fields are extracted
            if email_data.get("sender"):
                factors["sender"] = 0.8
            if email_data.get("subject"):
                factors["subject"] = 0.7
            if email_data.get("main_topic") and email_data.get("main_topic") != "Unknown":
                factors["topic"] = 0.9
            if email_data.get("sender_intent") and email_data.get("sender_intent") != "Unknown":
                factors["intent"] = 0.8
                
            # Default confidence if no factors
            if not factors:
                return 0.5
                
            return self.calculate_confidence_score(factors)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.3
    
    def _extract_basic_fields(self, content: str) -> Dict[str, Any]:
        """Extract basic email fields like sender, subject, etc."""
        fields = {
            "sender": None,
            "recipient": None,
            "subject": None,
            "body": content,
            "timestamp": datetime.now().isoformat()
        }
        
        lines = content.split('\n')
        header_section = True
        body_lines = []
        
        for line in lines:
            line = line.strip()
            
            if header_section:
                # Look for common email headers
                if line.lower().startswith('from:'):
                    fields["sender"] = self._extract_email_address(line)
                elif line.lower().startswith('to:'):
                    fields["recipient"] = self._extract_email_address(line)
                elif line.lower().startswith('subject:'):
                    fields["subject"] = line[8:].strip()
                elif line == "":
                    header_section = False
            else:
                body_lines.append(line)
        
        if body_lines:
            fields["body"] = '\n'.join(body_lines)
        
        return fields
    
    def _extract_email_address(self, text: str) -> Optional[str]:
        """Extract email address from text"""
        match = re.search(self.patterns["email"], text)
        return match.group(0) if match else None
    
    def _extract_entities(self, content: str) -> Dict[str, Any]:
        """Extract entities using regex patterns"""
        entities = {}
        
        # Extract emails
        emails = re.findall(self.patterns["email"], content)
        entities["mentioned_emails"] = list(set(emails))
        
        # Extract phone numbers
        phones = re.findall(self.patterns["phone"], content)
        entities["phone_numbers"] = list(set(phones))
        
        # Extract dates
        dates = re.findall(self.patterns["date"], content)
        entities["mentioned_dates"] = list(set(dates))
        
        # Extract monetary amounts
        amounts = re.findall(self.patterns["money"], content)
        entities["monetary_amounts"] = list(set(amounts))
        
        # Extract order/reference numbers
        order_matches = re.findall(self.patterns["order_number"], content, re.IGNORECASE)
        entities["reference_numbers"] = list(set(order_matches))
        
        return entities
    
    def _determine_urgency(self, content: str) -> str:
        """Determine email urgency based on content"""
        content_lower = content.lower()
        
        # Check for urgency keywords
        for urgency_level, keywords in self.urgency_patterns.items():
            for keyword in keywords:
                if keyword in content_lower:
                    return urgency_level
        
        # Default urgency
        return "medium"
    
    def _llm_extraction(self, content: str) -> Dict[str, Any]:
        """Use LLM for advanced entity extraction"""
        default_response = {
            "main_topic": "Unknown",
            "sender_intent": "Unknown", 
            "key_entities": [],
            "action_required": "Review required",
            "deadline_mentioned": None,
            "sentiment": "neutral"
        }
        
        try:
            # Check if LLM client is available
            if not self.llm_client:
                self.logger.warning("No LLM client available")
                return default_response
                
            # Limit content size for LLM
            content_preview = content[:1500] + "..." if len(content) > 1500 else content
            
            prompt = f"""
Analyze this email content and extract key information:

Email Content:
{content_preview}

Extract and provide the following information in JSON format:
{{
    "main_topic": "Brief description of the main topic",
    "sender_intent": "What the sender wants (request, complaint, inquiry, etc.)",
    "key_entities": ["Important names, companies, products mentioned"],
    "action_required": "What action is needed (if any)",
    "deadline_mentioned": "Any deadlines or time constraints mentioned",
    "sentiment": "positive, negative, or neutral"
}}

Respond with only the JSON object, no additional text.
"""
            
            response = self.llm_client.generate(prompt)
            
            if not response:
                self.logger.warning("LLM returned empty response")
                return default_response
                
            # Try to parse JSON response
            import json
            try:
                # Clean the response - remove any extra whitespace and potential markdown formatting
                cleaned_response = response.strip()
                
                # Remove markdown code blocks if present
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                
                cleaned_response = cleaned_response.strip()
                
                # Parse the JSON
                llm_data = json.loads(cleaned_response)
                
                # Ensure it's a dictionary (this is the key fix for your error)
                if isinstance(llm_data, dict):
                    # Validate required fields and fill in missing ones
                    for key, default_value in default_response.items():
                        if key not in llm_data:
                            llm_data[key] = default_value
                    return llm_data
                else:
                    self.logger.warning(f"LLM response is not a dictionary: {type(llm_data)}")
                    return default_response
                        
            except json.JSONDecodeError as e:
                self.logger.warning(f"LLM returned invalid JSON: {e}")
                self.logger.debug(f"Raw LLM response: {response}")
                
                # Try to extract some basic info from the text response
                basic_extraction = self._extract_basic_info_from_text(response)
                return {**default_response, **basic_extraction}
                
            except Exception as e:
                self.logger.error(f"Error processing LLM response: {e}")
                return default_response
        
        except Exception as e:
            self.logger.warning(f"LLM extraction failed: {e}")
            return default_response

    def _extract_basic_info_from_text(self, text: str) -> Dict[str, Any]:
        """Extract basic info from text when JSON parsing fails"""
        extracted = {}
        
        text_lower = text.lower()
        
        # Basic sentiment analysis
        if any(word in text_lower for word in ['happy', 'great', 'excellent', 'good', 'satisfied']):
            extracted['sentiment'] = 'positive'
        elif any(word in text_lower for word in ['angry', 'upset', 'terrible', 'bad', 'complaint']):
            extracted['sentiment'] = 'negative'
        else:
            extracted['sentiment'] = 'neutral'
            
        # Basic intent detection
        if any(word in text_lower for word in ['request', 'need', 'want', 'asking']):
            extracted['sender_intent'] = 'request'
        elif any(word in text_lower for word in ['question', 'ask', 'inquiry', 'wondering']):
            extracted['sender_intent'] = 'inquiry'
        elif any(word in text_lower for word in ['complaint', 'problem', 'issue', 'wrong']):
            extracted['sender_intent'] = 'complaint'
            
        return extracted

    def validate_extracted_data(self, input_data: Any) -> bool:
        """Validate email input data"""
        # Fix: Remove the super() call that's causing issues
        if input_data is None:
            self.logger.warning("Email input data is None")
            return False
            
        content = str(input_data).strip()
        if len(content) < 10:  # Too short to be meaningful
            self.logger.warning("Email content too short")
            return False
        
        return True
