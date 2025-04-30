"""
Compliance Manager

This script helps manage the process of bringing the codebase into compliance with myrules.mdc.
It provides tools for:
1. Analyzing current compliance state
2. Generating compliance reports
3. Suggesting refactoring steps
4. Tracking progress
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import json
import datetime
from rules_checker import RulesChecker

@dataclass
class ComplianceIssue:
    """Represents a single compliance issue."""
    file: str
    line: int
    issue_type: str
    description: str
    severity: str
    suggested_fix: str

class ComplianceManager:
    def __init__(self, project_root: Path):
        """Initialize the compliance manager."""
        self.project_root = project_root
        self.rules_checker = RulesChecker(project_root)
        self.issues: List[ComplianceIssue] = []
        self.progress_file = project_root / 'compliance_progress.json'
        
    def analyze_codebase(self) -> Dict[str, List[ComplianceIssue]]:
        """Perform a full analysis of the codebase."""
        violations = self.rules_checker.run_all_checks()
        categorized_issues = self._categorize_issues(violations)
        self._save_progress(categorized_issues)
        return categorized_issues
    
    def suggest_refactoring(self, file_path: Path) -> List[str]:
        """Suggest refactoring steps for a file."""
        suggestions = []
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Analyze file structure
        tree = ast.parse(content)
        
        # Check for large functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 50:  # Large function
                    suggestions.append(f"Consider splitting function {node.name} into smaller functions")
                    
        # Check for repeated code patterns
        # Add more sophisticated analysis here
        
        return suggestions
    
    def generate_progress_report(self) -> str:
        """Generate a markdown report of compliance progress."""
        if not self.progress_file.exists():
            return "No progress data available."
            
        with open(self.progress_file, 'r') as f:
            progress = json.load(f)
            
        report = ["# Compliance Progress Report\n"]
        report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for category, issues in progress.items():
            report.append(f"\n## {category}")
            report.append(f"Total Issues: {len(issues)}")
            report.append("Recent Fixes:")
            # Add progress tracking here
            
        return "\n".join(report)
    
    def _categorize_issues(self, violations: Dict[str, List[str]]) -> Dict[str, List[ComplianceIssue]]:
        """Categorize violations into structured issues."""
        categorized = {}
        for category, violation_list in violations.items():
            issues = []
            for violation in violation_list:
                # Parse violation string into structured data
                issue = self._parse_violation(violation, category)
                if issue:
                    issues.append(issue)
            categorized[category] = issues
        return categorized
    
    def _parse_violation(self, violation: str, category: str) -> Optional[ComplianceIssue]:
        """Parse a violation string into a ComplianceIssue."""
        try:
            if ':' in violation:
                file_path, rest = violation.split(':', 1)
                if ':' in rest:
                    line_no, description = rest.split(':', 1)
                    line_no = int(line_no)
                else:
                    line_no = 0
                    description = rest
            else:
                file_path = "unknown"
                line_no = 0
                description = violation
                
            return ComplianceIssue(
                file=file_path.strip(),
                line=line_no,
                issue_type=category,
                description=description.strip(),
                severity="high" if category in ["TYPE_HINTS", "DOCSTRINGS"] else "medium",
                suggested_fix=self._suggest_fix(category, description)
            )
        except Exception as e:
            print(f"Error parsing violation: {violation} - {str(e)}")
            return None
    
    def _suggest_fix(self, category: str, description: str) -> str:
        """Suggest a fix for a particular issue."""
        if category == "TYPE_HINTS":
            return "Add appropriate type hints to function parameters and return type"
        elif category == "DOCSTRINGS":
            return "Add docstring following Google style guide format"
        elif category == "FILE_LENGTH":
            return "Split file into smaller modules based on functionality"
        else:
            return "Review and fix according to style guide"
    
    def _save_progress(self, issues: Dict[str, List[ComplianceIssue]]) -> None:
        """Save current progress to file."""
        serializable = {
            category: [vars(issue) for issue in issue_list]
            for category, issue_list in issues.items()
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(serializable, f, indent=2)

def main():
    """Run the compliance manager."""
    project_root = Path(os.getcwd())
    manager = ComplianceManager(project_root)
    
    # Analyze codebase
    issues = manager.analyze_codebase()
    
    # Generate report
    report = manager.generate_progress_report()
    
    # Print summary
    print("\n=== Compliance Analysis Report ===\n")
    for category, issue_list in issues.items():
        print(f"\n{category} Issues: {len(issue_list)}")
        for issue in issue_list[:5]:  # Show first 5 issues in each category
            print(f"- {issue.file}:{issue.line} - {issue.description}")
            print(f"  Suggestion: {issue.suggested_fix}")
        if len(issue_list) > 5:
            print(f"  ... and {len(issue_list) - 5} more issues")
    
    # Save detailed report
    report_file = project_root / 'compliance_report.md'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to {report_file}")

if __name__ == '__main__':
    main() 