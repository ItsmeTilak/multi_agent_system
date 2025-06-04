import streamlit as st
import pandas as pd
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

# Check if we're running in Streamlit context
def is_streamlit_context():
    """Check if code is running in Streamlit context"""
    try:
        import streamlit as st
        return hasattr(st, 'session_state')
    except:
        return False

# Only import system components if we're in proper Streamlit context
if is_streamlit_context():
    try:
        from agents.classifier_agent import ClassifierAgent
        from agents.email_agent import EmailAgent
        from agents.json_agent import JSONAgent
        from agents.pdf_agent import PDFAgent
        from memory.database import DatabaseManager
        from utils.file_handler import FileHandler
        from config import Config
    except ImportError as e:
        st.error(f"Import error: {e}")
        st.stop()

class MultiAgentSystemUI:
    """Main UI class for the multi-agent system."""
    
    def __init__(self):
        """Initialize the UI components and agents."""
        if not is_streamlit_context():
            raise RuntimeError("MultiAgentSystemUI can only be initialized in Streamlit context")
        
        # Initialize with proper error handling
        try:
            self.db_manager = DatabaseManager()
            self.file_handler = FileHandler()
            
            # Pass db_manager to agents that need it
            self.classifier_agent = ClassifierAgent(self.db_manager)
            
            # Initialize specialized agents with db_manager
            self.agents = {
                'email': EmailAgent(self.db_manager),
                'json': JSONAgent(self.db_manager),
                'pdf': PDFAgent(self.db_manager)
            }
        except Exception as e:
            st.error(f"Failed to initialize system components: {e}")
            st.stop()
    
    @st.cache_data
    def get_system_status(_self):
        """Get system status with caching to avoid repeated DB calls"""
        try:
            records_count = _self.db_manager.get_total_records()
            return {"status": "connected", "records": records_count, "error": None}
        except Exception as e:
            return {"status": "error", "records": 0, "error": str(e)}
    
    def render_sidebar(self):
        """Render the sidebar with system information and controls."""
        with st.sidebar:
            st.title("🤖 Multi-Agent System")
            st.markdown("---")
            
            # System status
            st.subheader("System Status")
            status = self.get_system_status()
            
            if status["status"] == "connected":
                st.success(f"✅ Database Connected")
                st.info(f"📊 Total Records: {status['records']}")
            else:
                st.error(f"❌ Database Error: {status['error']}")
            
            # Clear database option
            st.markdown("---")
            st.subheader("🛠️ Admin Actions")
            
            if st.button("🗑️ Clear Database", type="secondary"):
                # Use a confirmation dialog
                if 'confirm_clear' not in st.session_state:
                    st.session_state.confirm_clear = False
                
                if not st.session_state.confirm_clear:
                    st.session_state.confirm_clear = st.checkbox("⚠️ Confirm deletion")
                
                if st.session_state.confirm_clear:
                    try:
                        self.db_manager.clear_all_records()
                        st.success("Database cleared!")
                        st.session_state.confirm_clear = False
                        # Clear cache to refresh status
                        self.get_system_status.clear()
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error clearing database: {e}")
    
    def render_file_upload(self):
        """Render the file upload section."""
        st.header("📁 File Upload & Processing")
        
        # File upload options
        upload_method = st.radio(
            "Choose input method:",
            ["Upload File", "Paste Text"],
            horizontal=True,
            key="upload_method"
        )
        
        uploaded_content = None
        file_type = None
        filename = None
        
        if upload_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['pdf', 'json', 'txt', 'eml'],
                help="Supported formats: PDF, JSON, TXT, EML",
                key="file_uploader"
            )
            
            if uploaded_file:
                filename = uploaded_file.name
                file_type = uploaded_file.type
                uploaded_content = uploaded_file.getvalue()
                
                # Display file info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("File Name", filename)
                with col2:
                    st.metric("File Type", file_type)
                with col3:
                    st.metric("File Size", f"{len(uploaded_content)} bytes")
                
        else:  # Paste Text
            text_input = st.text_area(
                "Paste your content here:",
                height=300,
                placeholder="Paste email content, JSON data, or any text...",
                key="text_input"
            )
            
            if text_input.strip():
                uploaded_content = text_input.encode('utf-8')
                filename = f"pasted_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                file_type = "text/plain"
                
                st.info(f"Content ready for processing ({len(uploaded_content)} bytes)")
        
        return uploaded_content, file_type, filename
    
    def process_content(self, content: bytes, file_type: str, filename: str):
        """Process the uploaded content through the agent system.""" 
        processing_result = None
        try:
            # Step 1: Classification
            st.subheader("🔍 Classification Results")
            
            with st.spinner("Classifying content..."):
                # For all files, decode bytes to string for classification
                if isinstance(content, bytes):
                    content_for_classify = content.decode('utf-8', errors='ignore')
                else:
                    content_for_classify = str(content)
                
                # Reformat filename to have timestamp before extension
                import os
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                classification_result = self.classifier_agent.classify(
                    content_for_classify, new_filename
                )
            
            # Display classification results
            if classification_result:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Format", classification_result.get('format', 'unknown'))
                with col2:
                    st.metric("Intent", classification_result.get('intent', 'unknown'))
                with col3:
                    st.metric("Confidence", f"{classification_result.get('confidence', 0):.2f}")
                
                # Step 2: Route to appropriate agent
                st.subheader("⚙️ Agent Processing")
                agent_type = classification_result.get('format', '').lower()
                
                # Fix: Map 'text' format to appropriate agent
                if agent_type == "text":
                    if file_type == "application/pdf":
                        agent_type = "pdf"
                    elif file_type == "text/plain":
                        agent_type = "email"
                
                if agent_type in self.agents:
                    with st.spinner(f"Processing with {agent_type.upper()} agent..."):
                        # For PDF and JSON files, pass bytes directly and classification result
                        import os
                        if agent_type in ["pdf", "json"]:
                            thread_id = str(uuid.uuid4())
                            name, ext = os.path.splitext(filename)
                            source_name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
                            processing_result = self.agents[agent_type].process(
                                content,
                                classification_result,
                                source_name=source_name,
                                file_size=len(content),
                                thread_id=thread_id
                            )
                        else:
                            thread_id = str(uuid.uuid4())
                            name, ext = os.path.splitext(filename)
                            source_name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
                            processing_result = self.agents[agent_type].process(
                                content_for_classify,
                                classification_result,
                                source_name=source_name,
                                file_size=len(content_for_classify.encode('utf-8')),
                                thread_id=thread_id
                            )
                
                else:
                    st.error(f"No agent available for format: {agent_type}")
                    return None
                
                # Display processing results
                if processing_result:
                    st.success(f"✅ Processed by {agent_type.upper()} Agent")
                    
                    # Show extracted data
                    extracted_fields = processing_result.get('extracted_fields', {})
                    if extracted_fields:
                        st.subheader("📋 Extracted Data")
                        
                        # Create a nice display format
                        if isinstance(extracted_fields, dict):
                            # Display in a more organized way
                            for key, value in extracted_fields.items():
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    st.write(f"**{key.replace('_', ' ').title()}:**")
                                with col2:
                                    st.write(value)
                        else:
                            st.json(extracted_fields)
                    
                    # Show confidence score from processing result
                    confidence_score = processing_result.get('confidence_score')
                    if confidence_score is not None:
                        st.metric("Calculated Confidence Score", f"{confidence_score:.2f}")
                    
                    # Show processing metadata
                    with st.expander("🔧 Processing Details"):
                        metadata = {
                            "Agent": agent_type,
                            "Status": processing_result.get('status', 'completed'),
                            "Error": processing_result.get('error_message', 'None'),
                            "Record ID": processing_result.get('record_id', 'N/A')
                        }
                        st.json(metadata)
                    
                    return processing_result
                else:
                    st.error("Agent processing returned no results")
                    return None
        except Exception as e:
            st.error(f"Error during content processing: {e}")
            return None
    
    def render_memory_logs(self):
        """Render the shared memory logs section."""
        st.header("📚 Memory Logs")
        
        try:
            # Get recent records with error handling
            records = self.db_manager.get_recent_records(limit=50) if hasattr(self.db_manager, 'get_recent_records') else []
            
            if records:
                # Convert to DataFrame for better display
                df = pd.DataFrame(records)
                
                # Remove processing_time column if present
                if 'processing_time' in df.columns:
                    df = df.drop(columns=['processing_time', 'sender', 'error_message'])
                
                # Format timestamp for display
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Decode source field if it appears as ASCII codes
                if 'source' in df.columns:
                    def decode_source(src):
                        if isinstance(src, str):
                            # Check if string looks like ASCII codes separated by commas
                            if all(part.isdigit() for part in src.split(',')):
                                try:
                                    chars = [chr(int(c)) for c in src.split(',')]
                                    return ''.join(chars)
                                except Exception:
                                    return src
                            else:
                                return src
                        else:
                            return str(src)
                    df['source'] = df['source'].apply(decode_source)
                
                # Convert extracted_fields column to string to avoid serialization errors
                if 'extracted_fields' in df.columns:
                    df['extracted_fields'] = df['extracted_fields'].apply(lambda x: json.dumps(x) if not isinstance(x, str) else x)
                
                # Display filters
                col1, col2 = st.columns(2)
                with col1:
                    format_options = ["All"] + (list(df['format'].unique()) if 'format' in df.columns else [])
                    format_filter = st.selectbox("Filter by Format:", format_options, key="format_filter")
                with col2:
                    intent_options = ["All"] + (list(df['intent'].unique()) if 'intent' in df.columns else [])
                    intent_filter = st.selectbox("Filter by Intent:", intent_options, key="intent_filter")
                
                # Apply filters
                filtered_df = df.copy()
                if format_filter != "All" and 'format' in df.columns:
                    filtered_df = filtered_df[filtered_df['format'] == format_filter]
                if intent_filter != "All" and 'intent' in df.columns:
                    filtered_df = filtered_df[filtered_df['intent'] == intent_filter]
                
                # Display the table
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show detailed view of selected record
                if len(filtered_df) > 0:
                    st.subheader("📄 Record Details")
                    selected_idx = st.selectbox(
                        "Select record to view details:",
                        range(len(filtered_df)),
                        format_func=lambda x: f"Record {x+1} - {filtered_df.iloc[x].get('source', 'Unknown')}",
                        key="record_selector"
                    )
                    
                    if selected_idx is not None:
                        selected_record = filtered_df.iloc[selected_idx]
                        
                        # Display record details in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Basic Info:**")
                            basic_info = [
                                ("ID", selected_record.get('id', 'N/A')),
                                ("Source", selected_record.get('source', 'N/A')),
                                ("Format", selected_record.get('format', 'N/A')),
                                ("Intent", selected_record.get('intent', 'N/A')),
                                ("Timestamp", selected_record.get('timestamp', 'N/A'))
                            ]
                            for label, value in basic_info:
                                st.write(f"- {label}: {value}")
                        
                        with col2:
                            st.write("**Processing Info:**")
                            processing_info = [
                                ("Processed By", selected_record.get('processed_by', 'N/A')),
                                ("Confidence", selected_record.get('confidence_score', 'N/A')),
                                ("Status", selected_record.get('status', 'N/A')),
                                ("Thread ID", selected_record.get('thread_id', 'N/A'))
                            ]
                            for label, value in processing_info:
                                st.write(f"- {label}: {value}")
                        
                        # Show extracted fields if available
                        extracted_fields = selected_record.get('extracted_fields')
                        if extracted_fields:
                            st.write("**Extracted Fields:**")
                            try:
                                if isinstance(extracted_fields, str):
                                    extracted_fields = json.loads(extracted_fields)
                                st.json(extracted_fields)
                            except Exception as e:
                                st.text(f"Raw data: {str(extracted_fields)}")
            else:
                st.info("No records found in the database.")
                st.write("Try processing some content first to see records here.")
                
        except Exception as e:
            st.error(f"Error loading memory logs: {str(e)}")
            st.write("This might be due to database connectivity issues or missing methods.")
    
    def render_statistics(self):
        """Render system statistics and analytics."""
        st.header("📊 System Statistics")
        
        try:
            # Check if statistics method exists
            if hasattr(self.db_manager, 'get_statistics'):
                stats = self.db_manager.get_statistics()
            else:
                # Create basic stats if advanced method doesn't exist
                total_records = self.db_manager.get_total_records() if hasattr(self.db_manager, 'get_total_records') else 0
                stats = {
                    'total_records': total_records,
                    'successful_processes': total_records,  # Assume all are successful for now
                    'failed_processes': 0,
                    'success_rate': 100.0 if total_records > 0 else 0.0
                }
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", stats.get('total_records', 0))
            with col2:
                st.metric("Successful Processes", stats.get('successful_processes', 0))
            with col3:
                st.metric("Failed Processes", stats.get('failed_processes', 0))
            with col4:
                success_rate = stats.get('success_rate', 0)
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            # Format distribution (if available)
            if stats.get('format_distribution'):
                st.subheader("📊 Format Distribution")
                format_df = pd.DataFrame(list(stats['format_distribution'].items()), 
                                       columns=['Format', 'Count'])
                st.bar_chart(format_df.set_index('Format'))
            
            # Intent distribution (if available)
            if stats.get('intent_distribution'):
                st.subheader("🎯 Intent Distribution")
                intent_df = pd.DataFrame(list(stats['intent_distribution'].items()), 
                                       columns=['Intent', 'Count'])
                st.bar_chart(intent_df.set_index('Intent'))
            
            # If no advanced stats available, show a message
            if not stats.get('format_distribution') and not stats.get('intent_distribution'):
                st.info("Advanced statistics will be available once you process more documents.")
            
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")

def main():
    """Main function to run the Streamlit app."""
    # Only run if we're in proper Streamlit context
    if not is_streamlit_context():
        print("Error: This module should only be run via 'streamlit run'")
        return
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="Multi-Agent System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS for larger tabs
    st.markdown(
        """
        <style>
        /* Increase font size and padding of tabs */
        [role="tablist"] [role="tab"] {
            font-size: 18px !important;
            padding: 12px 24px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Initialize the UI with error handling
    try:
        ui = MultiAgentSystemUI()
    except Exception as e:
        st.error(f"Failed to initialize UI: {e}")
        st.stop()
    
    # Render sidebar
    ui.render_sidebar()
    
    # Main content area
    st.title("🤖 Multi-Agent Document Processing System")
    st.markdown("Upload files or paste content to classify and extract structured data using specialized AI agents.")
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["📁 Process Content", "📚 Memory Logs"])
    
    with tab1:
        # File upload and processing
        uploaded_content, file_type, filename = ui.render_file_upload()
        
        if uploaded_content and st.button("🚀 Process Content", type="primary", key="process_button"):
            with st.container():
                processing_result = ui.process_content(uploaded_content, file_type, filename)
                
                if processing_result:
                    st.balloons()  # Celebration animation
                    st.success("Processing completed successfully!")
                    
                    # Offer to clear the input for next processing
                    if st.button("🔄 Process Another Document", key="clear_input"):
                        st.experimental_rerun()
    
    with tab2:
        ui.render_memory_logs()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Multi-Agent System v1.0 | Built with Streamlit & Python"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
