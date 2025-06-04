"""
Classifier Agent
Detects input format and classifies intent using LLM
"""

import re
import json
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .base_agent import BaseAgent

class ClassifierAgent(BaseAgent):
    """Agent for classifying file types and intents"""
    
    def __init__(self, db_manager=None, llm_client=None):
        super().__init__("classifier", llm_client, db_manager)
        
        # Define known intents and their patterns
        self.intent_patterns = {
            "rfq": ["request for quote", "rfq", "quotation request", "price inquiry"],
            "invoice": ["invoice", "bill", "payment due", "amount owed", "total due"],
            "complaint": ["complaint", "issue", "problem", "dissatisfied", "concern"],
            "regulation": ["regulation", "compliance", "policy", "guideline", "standard"],
            "contract": ["contract", "agreement", "terms", "conditions", "legal"],
            "order": ["purchase order", "order", "buy", "procurement"],
            "inquiry": ["inquiry", "question", "information", "help", "support"]
        }
    
    def validate_input(self, input_data: any) -> bool:
        """Validate input data for classification"""
        if input_data is None:
            return False
        if isinstance(input_data, (str, dict)):
            return bool(input_data)
        return False
    
    def process(self, input_data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Classify input format and intent
        
        Args:
            input_data: Can be file path, text content, or dict
            context: Optional context from previous processing
            
        Returns:
            Classification results with format, intent, and routing info
        """
        try:
            if not self.validate_input(input_data):
                return self.handle_error(ValueError("Invalid input data"))
            
            # Determine format and extract content
            format_type, content = self._detect_format(input_data)
            
            # Classify intent
            intent, confidence = self._classify_intent(content, format_type)
            
            # Determine routing
            target_agent = self._get_target_agent(format_type)
            
            result = {
                "success": True,
                "format": format_type,
                "intent": intent,
                "target_agent": target_agent,
                "confidence_score": confidence,
                "content_preview": self._get_content_preview(content),
                "agent": self.agent_name
            }
            
            # Log classification
            source = str(input_data) if isinstance(input_data, Path) else "direct_input"
            self.log_processing(
                source=source,
                format_type=format_type,
                intent=intent,
                extracted_fields=result,
                confidence_score=confidence
            )
            
            return result
            
        except Exception as e:
            return self.handle_error(e, "Classification failed")
    
    def _detect_format(self, input_data: Any) -> Tuple[str, str]:
        """
        Detect the format of input data
        
        Returns:
            Tuple of (format_type, content)
        """
        # Handle file path
        if isinstance(input_data, (str, Path)):
            file_path = Path(input_data)
            
            if file_path.exists():
                extension = file_path.suffix.lower()
                
                if extension == '.pdf':
                    return "PDF", str(file_path)
                elif extension in ['.json', '.jsonl']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return "JSON", content
                elif extension in ['.txt', '.eml']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return "EMAIL", content
            else:
                # Treat as direct text content
                content = str(input_data)
        
        # Handle direct content
        elif isinstance(input_data, dict):
            return "JSON", json.dumps(input_data)
        else:
            content = str(input_data)
        
        # Auto-detect format from content
        if self._is_json_content(content):
            return "JSON", content
        elif self._is_email_content(content):
            return "EMAIL", content
        else:
            return "TEXT", content
    
    def _is_json_content(self, content: str) -> bool:
        """Check if content is JSON format"""
        try:
            json.loads(content.strip())
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    
    def _is_email_content(self, content: str) -> bool:
        """Check if content looks like an email"""
        email_indicators = [
            r'from:.*?@.*?\.',
            r'to:.*?@.*?\.',
            r'subject:',
            r'dear\s+\w+',
            r'sincerely|regards|best\s+regards'
        ]
        
        content_lower = content.lower()
        matches = sum(1 for pattern in email_indicators 
                     if re.search(pattern, content_lower))
        
        return matches >= 2
    
    def _classify_intent(self, content: str, format_type: str) -> Tuple[str, float]:
        """
        Classify the intent of the content
        
        Returns:
            Tuple of (intent, confidence_score)
        """
        # First try rule-based classification
        rule_intent, rule_confidence = self._rule_based_classification(content)
        
        # If rule-based confidence is high, use it
        if rule_confidence > 0.7:
            return rule_intent, rule_confidence
        
        # Otherwise, use LLM classification
        llm_intent, llm_confidence = self._llm_classification(content, format_type)
        
        # Combine results (prefer LLM if available, fallback to rules)
        if llm_confidence > 0.5:
            return llm_intent, llm_confidence
        else:
            return rule_intent, rule_confidence
    
    def _rule_based_classification(self, content: str) -> Tuple[str, float]:
        """Rule-based intent classification"""
        content_lower = content.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in content_lower:
                    score += 1
            
            if score > 0:
                # Normalize score based on content length and pattern matches
                normalized_score = min(score / len(patterns), 1.0)
                intent_scores[intent] = normalized_score
        
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
            return best_intent, confidence
        
        return "unknown", 0.1
    
    def classify(self, content: str, filename: str = None, known_format: str = None) -> Dict[str, Any]:
        """
        Classify content with optional known format
        
        Args:
            content: Text content to classify
            filename: Optional filename for format detection
            known_format: Optional pre-determined format ('email', 'json', 'pdf')
            
        Returns:
            Classification results
        """
        try:
            # If format is known, use it; otherwise detect
            if known_format:
                format_type = known_format.upper()
                # Validate the known format
                if format_type not in ['EMAIL', 'JSON', 'PDF']:
                    self.logger.warning(f"Unknown format '{known_format}', falling back to auto-detection")
                    format_type, _ = self._detect_format(content)
            else:
                # Auto-detect format
                if filename:
                    # Try to detect from filename first
                    file_path = Path(filename)
                    extension = file_path.suffix.lower()
                    
                    if extension == '.pdf':
                        format_type = 'PDF'
                    elif extension in ['.json', '.jsonl']:
                        format_type = 'JSON'
                    elif extension in ['.txt', '.eml', '.msg']:
                        format_type = 'EMAIL'
                    else:
                        format_type, _ = self._detect_format(content)
                else:
                    format_type, _ = self._detect_format(content)
            
            # Classify intent
            intent, confidence = self._classify_intent(content, format_type)
            
            # Determine routing
            target_agent = self._get_target_agent(format_type)
            
            result = {
                "success": True,
                "format": format_type.lower(),
                "intent": intent,
                "target_agent": target_agent,
                "confidence": confidence,
                "content_preview": self._get_content_preview(content),
                "agent": self.agent_name
            }
            
            self.logger.info(f"Classification complete - Format: {format_type}, Intent: {intent}, Confidence: {confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return {
                "success": False,
                "format": "unknown",
                "intent": "unknown", 
                "target_agent": "email_agent",
                "confidence": 0.0,
                "error": str(e),
                "agent": self.agent_name
            }


    def _llm_classification(self, content: str, format_type: str) -> Tuple[str, float]:
        """LLM-based intent classification"""
        try:
            # Prepare content preview for LLM (limit size)
            content_preview = content[:1000] + "..." if len(content) > 1000 else content
            
            prompt = f"""
Analyze the following {format_type} content and classify its intent.

Content:
{content_preview}

Available intent categories:
- rfq: Request for Quote/Quotation
- invoice: Invoice or billing document
- complaint: Customer complaint or issue report
- regulation: Regulatory or compliance document
- contract: Legal contract or agreement
- order: Purchase order or procurement
- inquiry: General inquiry or question
- unknown: Cannot determine intent

Respond with only the intent category name and a confidence score (0.0-1.0) separated by a comma.
Example: "invoice,0.85"
"""
            
            response = self.llm_client.generate(prompt)

            if not response or not isinstance(response, str):
                self.logger.warning("Empty or invalid response from LLM.")
                return "unknown", 0.3

            if response and ',' in response:
                parts = response.strip().split(',')
                if len(parts) >= 2:
                    intent = parts[0].strip().lower()
                    try:
                        confidence = float(parts[1].strip())
                        # Validate intent is in known categories
                        if intent in self.intent_patterns or intent == "unknown":
                            return intent, min(max(confidence, 0.0), 1.0)
                    except ValueError:
                        pass
        
        except Exception as e:
            self.logger.warning(f"LLM classification failed: {e}")
        
        return "unknown", 0.3
    
    def _get_target_agent(self, format_type: str) -> str:
        """Determine which agent should process this format"""
        format_to_agent = {
            "PDF": "pdf_agent",
            "JSON": "json_agent", 
            "EMAIL": "email_agent",
            "TEXT": "email_agent"  # Default text to email agent
        }
        
        return format_to_agent.get(format_type, "email_agent")
    
    def _get_content_preview(self, content: str) -> str:
        """Get a preview of the content for display"""
        if len(content) <= 200:
            return content
        
        return content[:200] + "..."