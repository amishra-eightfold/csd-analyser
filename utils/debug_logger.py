"""Debug logging utilities for CSD Analyzer."""

import os
import logging
from datetime import datetime
from typing import Any, Dict, Optional
import streamlit as st
import json
from pathlib import Path

class DebugLogger:
    """Centralized debug logging functionality."""
    
    def __init__(self):
        """Initialize debug logger."""
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize log files
        self.app_log_file = self.log_dir / "app_debug.log"
        self.error_log_file = self.log_dir / "error_debug.log"
        self.api_log_file = self.log_dir / "api_debug.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set up file handlers
        self.setup_file_handler(self.app_log_file, "app")
        self.setup_file_handler(self.error_log_file, "error")
        self.setup_file_handler(self.api_log_file, "api")
        
        # Initialize debug container in session state
        if 'debug_container' not in st.session_state:
            st.session_state.debug_container = []
            
    def setup_file_handler(self, log_file: Path, logger_name: str):
        """Set up a file handler for a specific log file."""
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger = logging.getLogger(f"csd_analyzer.{logger_name}")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
    def log(self, message: str, data: Any = None, category: str = "app"):
        """
        Log a debug message with optional data.
        
        Args:
            message (str): Debug message
            data (Any, optional): Additional data to log
            category (str): Log category ("app", "error", or "api")
        """
        if not hasattr(st.session_state, 'debug_mode') or not st.session_state.debug_mode:
            return
            
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Format the debug message
        if data is not None:
            if isinstance(data, dict):
                data_str = "\n  " + "\n  ".join([f"{k}: {v}" for k, v in data.items()])
            else:
                data_str = str(data)
            log_message = f"[DEBUG] {timestamp} - {message}:{data_str}"
        else:
            log_message = f"[DEBUG] {timestamp} - {message}"
        
        # Get appropriate logger
        logger = logging.getLogger(f"csd_analyzer.{category}")
        
        # Log to file
        logger.debug(log_message)
        
        # Add to session state container
        if not hasattr(st.session_state, 'debug_container'):
            st.session_state.debug_container = []
        
        st.session_state.debug_container.append({
            'timestamp': timestamp,
            'category': category,
            'message': message,
            'data': data
        })
        
        # Keep only last 1000 messages
        if len(st.session_state.debug_container) > 1000:
            st.session_state.debug_container = st.session_state.debug_container[-1000:]
            
    def get_logs(self, category: Optional[str] = None) -> str:
        """
        Get logs as a formatted string.
        
        Args:
            category (Optional[str]): Filter logs by category
            
        Returns:
            str: Formatted log content
        """
        logs = []
        if category:
            log_file = getattr(self, f"{category}_log_file")
            if log_file.exists():
                logs.append(f"=== {category.upper()} LOGS ===")
                logs.append(log_file.read_text())
        else:
            for cat in ['app', 'error', 'api']:
                log_file = getattr(self, f"{cat}_log_file")
                if log_file.exists():
                    logs.append(f"=== {cat.upper()} LOGS ===")
                    logs.append(log_file.read_text())
                    logs.append("\n")
        
        return "\n".join(logs)
        
    def clear_logs(self, category: Optional[str] = None):
        """
        Clear log files and debug container.
        
        Args:
            category (Optional[str]): Clear specific category logs
        """
        if category:
            log_file = getattr(self, f"{category}_log_file")
            if log_file.exists():
                log_file.write_text("")
        else:
            for cat in ['app', 'error', 'api']:
                log_file = getattr(self, f"{cat}_log_file")
                if log_file.exists():
                    log_file.write_text("")
        
        st.session_state.debug_container = []
        
    def display_debug_ui(self):
        """Display debug UI with log viewer and controls."""
        if not st.session_state.debug_mode:
            return
            
        st.sidebar.markdown("---")
        st.sidebar.header("Debug Controls")
        
        # Log category filter
        category = st.sidebar.selectbox(
            "Log Category",
            ["All", "App", "Error", "API"],
            key="debug_category"
        )
        
        # Clear logs button
        if st.sidebar.button("Clear Logs"):
            self.clear_logs(category.lower() if category != "All" else None)
            st.success(f"Cleared {category} logs")
        
        # Download logs button
        log_content = self.get_logs(category.lower() if category != "All" else None)
        st.sidebar.download_button(
            "Download Logs",
            log_content,
            file_name=f"debug_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        # Display logs in main area
        with st.expander("Debug Log", expanded=True):
            if st.session_state.debug_container:
                for entry in st.session_state.debug_container:
                    if category == "All" or category.lower() == entry['category']:
                        st.code(
                            f"[{entry['timestamp']}] ({entry['category'].upper()}) {entry['message']}" +
                            (f"\n{json.dumps(entry['data'], indent=2)}" if entry['data'] else ""),
                            language="text"
                        )
            else:
                st.info("No debug messages yet.") 