"""Logger for OpenAI interactions."""

from typing import Dict, Any
from datetime import datetime
from pathlib import Path
import json
from config.logging_config import get_logger

class OpenAILogger:
    """
    Logger for OpenAI requests and responses.
    
    This class provides functionality to log OpenAI API interactions
    to both files and the application logging system.
    """
    
    def __init__(self, log_dir: str = "logs/openai") -> None:
        """
        Initialize the OpenAI logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger('openai')
    
    def log_interaction(self, request_data: Dict[str, Any], response_data: Dict[str, Any], interaction_type: str) -> None:
        """
        Log OpenAI interaction to file.
        
        Args:
            request_data: The request data sent to OpenAI
            response_data: The response data received from OpenAI
            interaction_type: Type of interaction (e.g., 'analysis', 'completion')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{interaction_type}_{timestamp}.json"
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "interaction_type": interaction_type,
            "request": request_data,
            "response": response_data
        }
        
        try:
            with open(self.log_dir / filename, 'w') as f:
                json.dump(log_data, f, indent=2)
            self.logger.info(f"Logged OpenAI interaction to {filename}")
            
            # Also save an HTML version for easy viewing
            html_filename = f"{interaction_type}_{timestamp}.html"
            self._create_html_log(log_data, html_filename)
        except Exception as e:
            self.logger.error(f"Failed to log OpenAI interaction: {str(e)}")
            
    def _create_html_log(self, log_data: Dict[str, Any], filename: str) -> None:
        """
        Create an HTML version of the log for easier viewing.
        
        Args:
            log_data: The log data to format
            filename: Output filename
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OpenAI Interaction Log</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                .timestamp {{ color: #888; font-style: italic; }}
                .section {{ margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>OpenAI Interaction Log</h1>
            <p class="timestamp">Timestamp: {log_data['timestamp']}</p>
            <p>Interaction Type: {log_data['interaction_type']}</p>
            
            <div class="section">
                <h2>Request</h2>
                <pre>{json.dumps(log_data['request'], indent=2)}</pre>
            </div>
            
            <div class="section">
                <h2>Response</h2>
                <pre>{json.dumps(log_data['response'], indent=2)}</pre>
            </div>
        </body>
        </html>
        """
        
        try:
            with open(self.log_dir / filename, 'w') as f:
                f.write(html_content)
        except Exception as e:
            self.logger.error(f"Failed to create HTML log: {str(e)}") 