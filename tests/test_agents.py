#!/usr/bin/env python3
"""
Unit Tests for Multi-Agent System
Tests all agents and core functionality of the system.
"""

import unittest
import os
import sys
import json
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import modules to test
from agents.base_agent import BaseAgent
from agents.classifier_agent import ClassifierAgent
from agents.email_agent import EmailAgent
from agents.json_agent import JSONAgent
from agents.pdf_agent import PDFAgent
from memory.database import DatabaseManager
from utils.file_handler import FileHandler
from utils.llm_client import llm_client 


class TestBaseAgent(unittest.TestCase):
    """Test the BaseAgent class."""
    
    class ConcreteAgent(BaseAgent):
        def __init__(self, name):
            super().__init__(name)
            self.name = name
        def process(self, input_data):
            return "processed"
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = self.ConcreteAgent("test_agent")
        
    def test_agent_initialization(self):
        """Test agent is initialized correctly."""
        self.assertEqual(self.agent.name, "test_agent")
        self.assertIsNotNone(self.agent.db_manager)
        
    def test_process_not_implemented(self):
        """Test that process method raises NotImplementedError."""
        with self.assertRaises(TypeError):
            BaseAgent("abstract_agent").process({})


class TestDatabaseManager(unittest.TestCase):
    """Test the DatabaseManager class."""
    
    def setUp(self):
        """Set up test database."""
        import tempfile
        import pathlib
        # Create temporary directory for tests
        self.temp_dir = pathlib.Path(tempfile.mkdtemp())
        # Create temporary database file in temp_dir
        self.temp_db = self.temp_dir / "test_database.db"
        
        # Initialize DatabaseManager with test database - it will create the correct schema
        self.db_manager = DatabaseManager(db_path=self.temp_db)
    
    def tearDown(self):
        """Clean up test database and directory."""
        import shutil
        if self.temp_db.exists():
            self.temp_db.unlink()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_store_interaction(self):
        """Test storing an interaction using create_record."""
        from memory.models import ProcessingRecord
        
        # Create a ProcessingRecord object
        record = ProcessingRecord(
            source='test_file.txt',
            format='email',
            intent='complaint',
            sender='test@example.com',
            urgency='high',
            extracted_fields='{"subject": "Test"}',
            processed_by='email_agent',
            confidence_score=0.95,
            status='completed'
        )
        
        record_id = self.db_manager.create_record(record)
        self.assertIsInstance(record_id, int)
        self.assertGreater(record_id, 0)
    
    def test_get_file_info(self):
        """Test getting file information."""
        import pathlib
        # Create test file path using pathlib.Path
        test_file = pathlib.Path(self.temp_dir) / "test.txt"
        # Write test content to file
        with open(test_file, 'w') as f:
            f.write("Test content")

        # Pass Path object to get_file_info
        info = self.file_handler.get_file_info(test_file)

        # Assert expected file info keys and values
        self.assertEqual(info['name'], 'test.txt')
        self.assertEqual(info['extension'], '.txt')
        self.assertGreater(info['size'], 0)
        self.assertIsInstance(info['modified'], datetime)


    
    def test_log_agent_action(self):
        """Test logging agent actions using create_record with agent action data."""
        from memory.models import ProcessingRecord
        
        # Create an agent action record
        action_record = ProcessingRecord(
            source='test.txt',
            format='email',
            processed_by='test_agent',
            status='processing',
            processing_time=0.5,
            extracted_fields='{"action": "test_action", "details": "Test details"}'
        )
        
        record_id = self.db_manager.create_record(action_record)
        retrieved = self.db_manager.get_record(record_id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.processed_by, 'test_agent')
        self.assertEqual(retrieved.processing_time, 0.5)
    
    def test_update_record(self):
        """Test updating a record."""
        from memory.models import ProcessingRecord
        
        # Create initial record
        record = ProcessingRecord(
            source='test_file.txt',
            format='email',
            status='pending'
        )
        
        record_id = self.db_manager.create_record(record)
        
        # Update the record
        record.id = record_id
        record.status = 'completed'
        record.confidence_score = 0.95
        
        updated = self.db_manager.update_record(record)
        self.assertTrue(updated)
        
        # Verify update
        retrieved = self.db_manager.get_record(record_id)
        self.assertEqual(retrieved.status, 'completed')
        self.assertEqual(retrieved.confidence_score, 0.95)
    
    def test_get_records_by_status(self):
        """Test retrieving records by status."""
        from memory.models import ProcessingRecord
        
        # Create test records with different statuses
        record1 = ProcessingRecord(source='test1.txt', format='email', status='pending')
        record2 = ProcessingRecord(source='test2.txt', format='pdf', status='completed')
        record3 = ProcessingRecord(source='test3.txt', format='json', status='pending')
        
        self.db_manager.create_record(record1)
        self.db_manager.create_record(record2)
        self.db_manager.create_record(record3)
        
        # Get pending records
        pending_records = self.db_manager.get_records_by_status('pending')
        self.assertEqual(len(pending_records), 2)
        
        # Get completed records
        completed_records = self.db_manager.get_records_by_status('completed')
        self.assertEqual(len(completed_records), 1)
    
    def test_get_recent_records(self):
        """Test retrieving recent records."""
        from memory.models import ProcessingRecord
        
        # Create test records
        for i in range(5):
            record = ProcessingRecord(
                source=f'test{i}.txt',
                format='email',
                status='completed'
            )
            self.db_manager.create_record(record)
        
        # Get recent records
        recent = self.db_manager.get_recent_records(limit=3)
        self.assertEqual(len(recent), 3)
        
        # Should be in descending order by timestamp
        timestamps = [record.timestamp for record in recent]
        self.assertEqual(timestamps, sorted(timestamps, reverse=True))
    
    def test_search_records(self):
        """Test searching records."""
        from memory.models import ProcessingRecord
        
        # Create test records
        record1 = ProcessingRecord(
            source='important_email.txt',
            format='email',
            intent='complaint',
            sender='john@example.com'
        )
        record2 = ProcessingRecord(
            source='invoice.pdf',
            format='pdf',
            intent='billing',
            extracted_fields='{"customer": "john"}'
        )
        
        self.db_manager.create_record(record1)
        self.db_manager.create_record(record2)
        
        # Search for "john"
        results = self.db_manager.search_records('john')
        self.assertEqual(len(results), 2)
        
        # Search for "complaint"
        results = self.db_manager.search_records('complaint')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].intent, 'complaint')
    
    def test_get_statistics(self):
        """Test getting database statistics."""
        from memory.models import ProcessingRecord
        
        # Create test records with various attributes
        records = [
            ProcessingRecord(format='email', status='completed', intent='complaint', processing_time=1.5),
            ProcessingRecord(format='pdf', status='pending', intent='billing', processing_time=2.0),
            ProcessingRecord(format='email', status='completed', intent='inquiry', processing_time=1.0),
        ]
        
        for record in records:
            self.db_manager.create_record(record)
        
        stats = self.db_manager.get_statistics()
        
        self.assertEqual(stats['total_records'], 3)
        self.assertEqual(stats['by_status']['completed'], 2)
        self.assertEqual(stats['by_status']['pending'], 1)
        self.assertEqual(stats['by_format']['email'], 2)
        self.assertEqual(stats['by_format']['pdf'], 1)
        self.assertGreater(stats['avg_processing_time'], 0)
    
    def test_delete_record(self):
        """Test deleting a record."""
        from memory.models import ProcessingRecord
        
        # Create test record
        record = ProcessingRecord(
            source='test_delete.txt',
            format='email'
        )
        
        record_id = self.db_manager.create_record(record)
        
        # Verify record exists
        retrieved = self.db_manager.get_record(record_id)
        self.assertIsNotNone(retrieved)
        
        # Delete record
        deleted = self.db_manager.delete_record(record_id)
        self.assertTrue(deleted)
        
        # Verify record is gone
        retrieved = self.db_manager.get_record(record_id)
        self.assertIsNone(retrieved)
    
    def test_get_total_records(self):
        """Test getting total record count."""
        from memory.models import ProcessingRecord
        
        initial_count = self.db_manager.get_total_records()
        
        # Add some records
        for i in range(3):
            record = ProcessingRecord(
                source=f'test{i}.txt',
                format='email'
            )
            self.db_manager.create_record(record)
        
        final_count = self.db_manager.get_total_records()
        self.assertEqual(final_count, initial_count + 3)
    
    def test_get_records_by_thread(self):
        """Test retrieving records by thread ID."""
        from memory.models import ProcessingRecord
        
        thread_id = "thread_123"
        
        # Create records with same thread ID
        record1 = ProcessingRecord(source='test1.txt', format='email', thread_id=thread_id)
        record2 = ProcessingRecord(source='test2.txt', format='pdf', thread_id=thread_id)
        record3 = ProcessingRecord(source='test3.txt', format='json', thread_id="different_thread")
        
        self.db_manager.create_record(record1)
        self.db_manager.create_record(record2)
        self.db_manager.create_record(record3)
        
        # Get records by thread
        thread_records = self.db_manager.get_records_by_thread(thread_id)
        self.assertEqual(len(thread_records), 2)
        
        # Should be in ascending order by timestamp
        timestamps = [record.timestamp for record in thread_records]
        self.assertEqual(timestamps, sorted(timestamps))

class TestFileHandler(unittest.TestCase):
    """Test the FileHandler class."""
    
    def setUp(self):
        """Set up test files."""
        import pathlib
        self.temp_dir = pathlib.Path(tempfile.mkdtemp())
        self.file_handler = FileHandler(upload_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_uploaded_file(self):
        """Test saving uploaded files."""
        # Create mock uploaded file
        mock_file = Mock()
        mock_file.name = "test.txt"
        mock_file.read.return_value = b"Test content"
        
        file_path = self.file_handler.save_uploaded_file(mock_file)
        
        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(file_path.endswith("test.txt"))
        
        with open(file_path, 'rb') as f:
            content = f.read()
            self.assertEqual(content, b"Test content")
    
    def test_get_file_info(self):
        """Test getting file information."""
        import pathlib
        # Create test file path using pathlib.Path
        test_file = pathlib.Path(self.temp_dir) / "test.txt"
        # Write test content to file
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        # Pass Path object to get_file_info
        info = self.file_handler.get_file_info(test_file)
        
        self.assertEqual(info['name'], 'test.txt')
        self.assertEqual(info['extension'], '.txt')
        self.assertGreater(info['size'], 0)
        self.assertIsInstance(info['modified'], (str, type(None)))


class TestLLMClient(unittest.TestCase):
    """Test the LLMClient class."""
    
    def setUp(self):
        """Set up LLM client."""
        self.llm_client = llm_client()

    @patch('requests.post')
    def test_generate_response_success(self, mock_post):
        """Test successful LLM response generation."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Test response'}}]
        }
        mock_post.return_value = mock_response
        
        response = self.llm_client.generate_response("Test prompt")
        
        self.assertEqual(response, "Test response")
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_generate_response_failure(self, mock_post):
        """Test LLM response generation failure."""
        # Mock failed API response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response
        
        response = self.llm_client.generate_response("Test prompt")
        
        self.assertIsNone(response)


class TestClassifierAgent(unittest.TestCase):
    """Test the ClassifierAgent class."""
    
    def setUp(self):
        """Set up classifier agent."""
        self.agent = ClassifierAgent()
    
    @patch.object(ClassifierAgent, '_call_llm')
    def test_classify_email(self, mock_llm):
        """Test email classification."""
        mock_llm.return_value = "format: email\nintent: complaint\nconfidence: 0.95"
        
        input_data = {
            'content': 'Subject: Product Issue\n\nI am having problems with my order...',
            'file_path': 'test_email.txt'
        }
        
        result = self.agent.process(input_data)
        
        self.assertEqual(result['format'], 'email')
        self.assertEqual(result['intent'], 'complaint')
        self.assertGreater(result['confidence_score'], 0.9)
    
    @patch.object(ClassifierAgent, '_call_llm')
    def test_classify_json(self, mock_llm):
        """Test JSON classification."""
        mock_llm.return_value = "format: json\nintent: invoice\nconfidence: 0.98"
        
        input_data = {
            'content': '{"invoice_id": "123", "amount": 100.00}',
            'file_path': 'test_invoice.json'
        }
        
        result = self.agent.process(input_data)
        
        self.assertEqual(result['format'], 'json')
        self.assertEqual(result['intent'], 'invoice')
        self.assertGreater(result['confidence'], 0.9)
    
    def test_detect_format_by_extension(self):
        """Test format detection by file extension."""
        test_cases = [
            ('test.pdf', 'pdf'),
            ('test.json', 'json'),
            ('test.txt', 'email'),
            ('test.eml', 'email'),
        ]
        
        for filename, expected_format in test_cases:
            detected_format = self.agent._detect_format_by_extension(filename)
            self.assertEqual(detected_format, expected_format)


class TestEmailAgent(unittest.TestCase):
    """Test the EmailAgent class."""
    
    def setUp(self):
        """Set up email agent."""
        self.agent = EmailAgent()
    
    @patch.object(EmailAgent, '_call_llm')
    def test_process_email(self, mock_llm):
        """Test email processing."""
        mock_llm.return_value = '''
        sender: john.doe@example.com
        subject: Product Issue
        urgency: high
        intent: complaint
        extracted_data: {"product_id": "ABC123", "issue_type": "defective"}
        '''
        
        input_data = {
            'content': 'From: john.doe@example.com\nSubject: Product Issue\n\nI have a problem...',
            'source': 'test_email.txt'
        }
        
        result = self.agent.process(input_data)
        
        self.assertEqual(result['sender'], 'john.doe@example.com')
        self.assertEqual(result['urgency'], 'high')
        self.assertEqual(result['intent'], 'complaint')
        self.assertIn('product_id', result['extracted_fields'])
    
    def test_extract_basic_email_info(self):
        """Test basic email information extraction."""
        email_content = '''
        From: sender@example.com
        To: recipient@example.com
        Subject: Test Subject
        Date: Mon, 1 Jan 2024 12:00:00 +0000
        
        This is the email body.
        '''
        
        info = self.agent._extract_basic_email_info(email_content)
        
        self.assertEqual(info['sender'], 'sender@example.com')
        self.assertEqual(info['subject'], 'Test Subject')
        self.assertIn('This is the email body', info['body'])


class TestJSONAgent(unittest.TestCase):
    """Test the JSONAgent class."""
    
    def setUp(self):
        """Set up JSON agent."""
        self.agent = JSONAgent()
    
    def test_process_valid_json(self):
        """Test processing valid JSON."""
        input_data = {
            'content': '{"invoice_id": "INV-123", "amount": 1500.00, "customer": "John Doe"}',
            'source': 'test_invoice.json'
        }
        
        result = self.agent.process(input_data)
        
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['is_valid'])
        self.assertIn('invoice_id', result['extracted_fields'])
        self.assertEqual(result['extracted_fields']['invoice_id'], 'INV-123')
    
    def test_process_invalid_json(self):
        """Test processing invalid JSON."""
        input_data = {
            'content': '{"invalid": json, "missing": quotes}',
            'source': 'test_invalid.json'
        }
        
        result = self.agent.process(input_data)
        
        self.assertEqual(result['status'], 'error')
        self.assertFalse(result['is_valid'])
        self.assertIsNotNone(result['error_message'])
    
    def test_validate_schema(self):
        """Test JSON schema validation."""
        valid_json = {"name": "test", "value": 123}
        invalid_json = {"missing_required_field": True}
        
        # Test with valid JSON
        is_valid, errors = self.agent._validate_schema(valid_json)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)


class TestPDFAgent(unittest.TestCase):
    """Test the PDFAgent class."""
    
    def setUp(self):
        """Set up PDF agent."""
        self.agent = PDFAgent()
    
    def test_process_with_mock_content(self):
        """Test PDF processing with mocked content."""
        # Mock PDF extraction
        with patch.object(self.agent, '_extract_pdf_text') as mock_extract:
            mock_extract.return_value = "This is extracted PDF text content."
            
            with patch.object(self.agent, '_call_llm') as mock_llm:
                mock_llm.return_value = '''
                document_type: complaint
                extracted_data: {"complaint_id": "C-123", "issue": "service quality"}
                key_entities: ["complaint", "service", "quality"]
                '''
                
                input_data = {
                    'file_path': 'test_complaint.pdf',
                    'source': 'test_complaint.pdf'
                }
                
                result = self.agent.process(input_data)
                
                self.assertEqual(result['status'], 'completed')
                self.assertIn('complaint_id', result['extracted_fields'])
                self.assertEqual(result['document_type'], 'complaint')
    
    def test_extract_pdf_text_file_not_found(self):
        """Test PDF text extraction with missing file."""
        text = self.agent._extract_pdf_text('nonexistent.pdf')
        self.assertEqual(text, '')


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Initialize test database
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT, format TEXT, intent TEXT, timestamp TEXT,
                sender TEXT, urgency TEXT, extracted_fields TEXT,
                processed_by TEXT, thread_id TEXT, confidence_score REAL,
                status TEXT DEFAULT 'pending', error_message TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER, agent_name TEXT, action TEXT,
                details TEXT, timestamp TEXT, execution_time REAL,
                success BOOLEAN DEFAULT TRUE
            )
        ''')
        conn.commit()
        conn.close()
    
    def tearDown(self):
        """Clean up integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
        os.unlink(self.temp_db.name)
    
    @patch('agents.classifier_agent.ClassifierAgent._call_llm')
    @patch('agents.email_agent.EmailAgent._call_llm')
    def test_email_processing_workflow(self, mock_email_llm, mock_classifier_llm):
        """Test complete email processing workflow."""
        # Mock LLM responses
        mock_classifier_llm.return_value = "format: email\nintent: complaint\nconfidence: 0.95"
        mock_email_llm.return_value = '''
        sender: test@example.com
        subject: Issue Report
        urgency: high
        intent: complaint
        extracted_data: {"issue_type": "product_defect"}
        '''
        
        # Initialize agents
        classifier = ClassifierAgent()
        email_agent = EmailAgent()
        
        # Simulate workflow
        input_data = {
            'content': 'From: test@example.com\nSubject: Issue Report\n\nI have an issue...',
            'file_path': 'test_email.txt'
        }
        
        # Step 1: Classification
        classification_result = classifier.process(input_data)
        self.assertEqual(classification_result['format'], 'email')
        self.assertEqual(classification_result['intent'], 'complaint')
        
        # Step 2: Email processing
        input_data['source'] = 'test_email.txt'
        email_result = email_agent.process(input_data)
        self.assertEqual(email_result['sender'], 'test@example.com')
        self.assertEqual(email_result['urgency'], 'high')


def run_tests():
    """Run all tests."""
    print("🧪 Running Multi-Agent System Tests...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBaseAgent,
        TestDatabaseManager,
        TestFileHandler,
        TestLLMClient,
        TestClassifierAgent,
        TestEmailAgent,
        TestJSONAgent,
        TestPDFAgent,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"🏁 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Error:')[-1].strip()}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)