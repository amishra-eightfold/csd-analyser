"""Debug logging utilities for CSD Analyzer."""

import os
import logging
from datetime import datetime
from typing import Any, Dict, Optional, List
import streamlit as st
import json
from pathlib import Path
import pandas as pd
import numpy as np

class DebugLogger:
    """Centralized debug logging with UI display capabilities."""
    
    def __init__(self, max_entries=1000):
        self.log_entries = []
        self.max_entries = max_entries
        
    def _sanitize_data(self, data):
        """Sanitize data to ensure it's JSON serializable."""
        if data is None:
            return None
            
        # Handle numpy data types
        if isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(data)
        if isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
            return float(data)
        if isinstance(data, np.bool_):
            return bool(data)
        if isinstance(data, np.ndarray):
            return self._sanitize_data(data.tolist())
            
        if isinstance(data, (str, int, float, bool)):
            return data
            
        if isinstance(data, (datetime, pd.Timestamp)):
            return str(data)
            
        if isinstance(data, pd.Series):
            return self._sanitize_data(data.to_dict())
            
        if isinstance(data, pd.DataFrame):
            return {
                'shape': self._sanitize_data(data.shape),
                'columns': list(data.columns),
                'sample': self._sanitize_data(data.head().to_dict()) if not data.empty else {}
            }
            
        if isinstance(data, (list, tuple)):
            return [self._sanitize_data(item) for item in data]
            
        if isinstance(data, dict):
            return {
                str(k): self._sanitize_data(v)
                for k, v in data.items()
            }
            
        if isinstance(data, (pd.Period, pd.Interval)):
            return str(data)
            
        if hasattr(data, '__dict__'):
            return self._sanitize_data(data.__dict__)
            
        # For any other type, convert to string
        try:
            return str(data)
        except:
            return f"<non-serializable object of type {type(data).__name__}>"
    
    def log(self, message: str, data: Any = None, category: str = "info"):
        """Add a log entry with optional data and category."""
        try:
            sanitized_data = self._sanitize_data(data)
            
            entry = {
                'timestamp': datetime.now().isoformat(),
                'message': str(message),
                'data': sanitized_data,
                'category': category
            }
            
            self.log_entries.append(entry)
            
            # Trim log if it exceeds max entries
            if len(self.log_entries) > self.max_entries:
                self.log_entries = self.log_entries[-self.max_entries:]
                
        except Exception as e:
            # Fallback logging if something goes wrong
            fallback_entry = {
                'timestamp': datetime.now().isoformat(),
                'message': f"Error logging message: {str(message)}. Error: {str(e)}",
                'data': None,
                'category': 'error'
            }
            self.log_entries.append(fallback_entry)
    
    def get_logs(self, category: Optional[str] = None) -> List[Dict]:
        """Get all logs, optionally filtered by category."""
        if category:
            return [entry for entry in self.log_entries if entry['category'] == category]
        return self.log_entries
    
    def clear_logs(self):
        """Clear all log entries."""
        self.log_entries = []
    
    def display_debug_ui(self):
        """Display debug information in the Streamlit UI."""
        try:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Debug Logs")
            
            # Add log filtering options
            categories = list(set(entry['category'] for entry in self.log_entries))
            selected_category = st.sidebar.selectbox(
                "Filter by category",
                ["All"] + categories
            )
            
            # Add search box
            search_term = st.sidebar.text_input("Search logs", "").lower()
            
            # Filter logs
            filtered_logs = self.log_entries
            if selected_category != "All":
                filtered_logs = [
                    entry for entry in filtered_logs 
                    if entry['category'] == selected_category
                ]
            
            if search_term:
                filtered_logs = [
                    entry for entry in filtered_logs
                    if search_term in entry['message'].lower() or
                    (entry['data'] and search_term in str(entry['data']).lower())
                ]
            
            # Display log count
            st.sidebar.markdown(f"Showing {len(filtered_logs)} of {len(self.log_entries)} logs")
            
            # Add clear logs button
            if st.sidebar.button("Clear Logs"):
                self.clear_logs()
                st.sidebar.success("Logs cleared!")
                return
            
            # Display logs
            for i, entry in enumerate(reversed(filtered_logs)):
                try:
                    # Create a unique label for each expander
                    expander_label = f"[{entry['category'].upper()}] {entry['message'][:50]}..."
                    
                    with st.sidebar.expander(expander_label):
                        st.markdown(f"**Time:** {entry['timestamp']}")
                        st.markdown(f"**Category:** {entry['category']}")
                        st.markdown("**Message:**")
                        st.markdown(entry['message'])
                        
                        if entry['data']:
                            st.markdown("**Data:**")
                            try:
                                # Try to display as JSON
                                st.code(json.dumps(entry['data'], indent=2), language='json')
                            except:
                                # Fallback to string representation
                                st.code(str(entry['data']))
                                
                except Exception as e:
                    st.sidebar.error(f"Error displaying log entry: {str(e)}")
                    
        except Exception as e:
            st.sidebar.error(f"Error in debug UI: {str(e)}")
            if hasattr(st, 'session_state') and st.session_state.get('debug_mode'):
                st.sidebar.exception(e) 