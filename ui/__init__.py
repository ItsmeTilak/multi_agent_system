"""
UI module for the multi-agent system.

This module contains the Streamlit-based user interface components
for file upload, agent interaction, and result visualization.
"""

from .streamlit_app import main as run_streamlit_app

__all__ = ['run_streamlit_app']