#!/usr/bin/env python3
"""
Multi-Agent AI System Entry Point
Main script to initialize and run the system in CLI or UI mode
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from memory.database import DatabaseManager

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import LOGGING_CONFIG, DATABASE_URL, UPLOADS_DIR

def setup_logging():
    """Configure logging based on LOGGING_CONFIG"""
    logging.basicConfig(
        level=LOGGING_CONFIG["level"],
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.FileHandler(LOGGING_CONFIG["file"]),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized.")

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        UPLOADS_DIR,
        Path(DATABASE_URL).parent,
        Path(LOGGING_CONFIG["file"]).parent
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory ensured: {directory}")

def initialize_database():
    """Initialize the SQLite database"""
    try:
        db_manager = DatabaseManager()
        logging.info("Database initialized successfully")
        return db_manager
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")
        raise

def run_cli_mode(args):
    """Run the system in CLI mode"""
    from agents.classifier_agent import ClassifierAgent
    from agents.email_agent import EmailAgent
    from agents.json_agent import JSONAgent
    from agents.pdf_agent import PDFAgent
    from utils.file_handler import FileHandler
    
    print(f"📄 Processing file: {args.file}")
    print(f"📋 Format: {args.format if args.format else 'auto-detect'}")
    
    # Initialize system components
    db_manager = initialize_database()
    file_handler = FileHandler()
    
    # Initialize agents
    classifier = ClassifierAgent(db_manager)
    agents = {
        'email': EmailAgent(db_manager),
        'json': JSONAgent(db_manager),
        'pdf': PDFAgent(db_manager)
    }
    
    try:
        # Read and process file
        if not os.path.exists(args.file):
            print(f"❌ Error: File not found: {args.file}")
            return
        
        file_content = file_handler.read_file(args.file)
        file_name = os.path.basename(args.file)
        
        # Classify file
        print("🔍 Classifying file...")
        # classification = classifier.classify(file_content, file_name, known_format=args.format)
        
        if args.format:
            # If format is provided, we can skip classification or use it as a hint
            
            print(f"📋 Using provided format: {args.format}")
            classification = classifier.classify(file_content, file_name, known_format=args.format)
        else:
            # Auto-detect format using classifier
            classification = classifier.classify(file_content, file_name)

        print(f"✅ Classification Results:")
        print(f"   Format: {classification.get('format', 'unknown')}")
        print(f"   Intent: {classification.get('intent', 'unknown')}")
        print(f"   Confidence: {classification.get('confidence', 0):.2f}")
        
        # Route to appropriate agent
        file_format = classification.get('format', 'unknown')
        if file_format in agents:
            print(f"🤖 Processing with {file_format.title()} Agent...")
            if file_format == 'pdf':
                # PDF Agent expects file path, not content
                result = agents[file_format].process(args.file, classification)
            elif file_format == 'json':
                # JSON Agent expects file path, not content
                result = agents[file_format].process(args.file, classification)
            else:
                # Other agents might expect content
                result = agents[file_format].process(file_content, classification)
            
            print(f"📊 Extraction Results:")
            for key, value in result.get('extracted_fields', {}).items():
                print(f"   {key}: {value}")
        else:
            print(f"❌ No agent available for format: {file_format}")
            
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        logging.error(f"CLI processing error: {e}")

def run_ui_mode():
    """Run the system in UI mode"""
    from ui.streamlit_app import main as run_streamlit
    print("🌐 Starting Streamlit UI...")
    run_streamlit()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent AI Document Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run web UI
  python main.py --file email.txt --format email   # Process email in CLI
  python main.py --file invoice.json               # Auto-detect and process
  python main.py --no-ui                           # CLI mode without file
        """
    )
    
    parser.add_argument(
        '--file', 
        type=str, 
        help='Path to file to process (enables CLI mode)'
    )
    
    parser.add_argument(
        '--format', 
        type=str, 
        choices=['email', 'json', 'pdf'],
        help='Force specific file format (optional)'
    )
    
    parser.add_argument(
        '--no-ui', 
        action='store_true',
        help='Run in CLI mode without processing any file'
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    print("🚀 Starting Multi-Agent AI System...")

    # Setup logging
    setup_logging()
    
    # Create directories
    setup_directories()
    
    print("✅ System initialization complete!")
    
    # Determine mode based on arguments
    if args.file or args.no_ui:
        # CLI mode
        if args.file:
            run_cli_mode(args)
        else:
            print("💻 CLI mode active. Use --file to process a document.")
    else:
        # UI mode (default)
        run_ui_mode()

if __name__ == "__main__":
    main()