"""
Rules Compliance Checker

This script helps enforce myrules.mdc compliance by:
1. Checking code style and structure
2. Verifying documentation
3. Analyzing test coverage
4. Detecting potential duplications
5. Validating performance metrics
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytest
import pylint.lint
import coverage
import docstring_parser
import time
import chardet

class RulesChecker:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.rules_file = project_root / '.cursor' / 'rules' / 'myrules.mdc'
        self.violations: List[str] = []
        
    def read_file_content(self, file_path: Path) -> str:
        """Read file content with automatic encoding detection."""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        # Detect the encoding
        result = chardet.detect(raw_data)
        encoding = result['encoding'] if result['encoding'] else 'utf-8'
        
        try:
            return raw_data.decode(encoding)
        except UnicodeDecodeError:
            # Fallback to latin-1 if detection fails
            return raw_data.decode('latin-1')
        
    def check_file_length(self, file_path: Path) -> List[str]:
        """Check if file exceeds 500 lines."""
        violations = []
        try:
            content = self.read_file_content(file_path)
            lines = content.splitlines()
            if len(lines) > 500:
                violations.append(f"{file_path} exceeds 500 lines (current: {len(lines)})")
        except Exception as e:
            violations.append(f"Could not check file length for {file_path}: {str(e)}")
        return violations

    def check_type_hints(self, file_path: Path) -> List[str]:
        """Verify presence of type hints in functions."""
        violations = []
        try:
            content = self.read_file_content(file_path)
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check return type annotation
                    if node.returns is None:
                        violations.append(f"{file_path}:{node.lineno} - Missing return type hint in function {node.name}")
                    # Check parameter type annotations
                    for arg in node.args.args:
                        if arg.annotation is None:
                            violations.append(f"{file_path}:{node.lineno} - Missing type hint for parameter {arg.arg} in function {node.name}")
        except Exception as e:
            violations.append(f"Could not check type hints for {file_path}: {str(e)}")
        return violations

    def check_docstrings(self, file_path: Path) -> List[str]:
        """Verify presence and quality of docstrings."""
        violations = []
        try:
            content = self.read_file_content(file_path)
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if ast.get_docstring(node) is None:
                        violations.append(f"{file_path}:{node.lineno} - Missing docstring in {node.name}")
        except Exception as e:
            violations.append(f"Could not check docstrings for {file_path}: {str(e)}")
        return violations

    def check_test_coverage(self) -> Tuple[float, List[str]]:
        """Run tests and verify coverage meets 80% threshold."""
        violations = []
        try:
            cov = coverage.Coverage()
            cov.start()
            pytest.main(['--quiet'])
            cov.stop()
            cov.save()
            
            total = cov.report()
            if total < 80:
                violations.append(f"Test coverage ({total}%) is below required 80%")
            return total, violations
        except Exception as e:
            return 0.0, [f"Could not check test coverage: {str(e)}"]

    def check_performance(self, func, test_data) -> List[str]:
        """Verify function performance meets requirements."""
        violations = []
        try:
            start_time = time.time()
            func(test_data)
            execution_time = time.time() - start_time
            
            # Check against PRD performance requirements
            if 'query' in func.__name__ and execution_time > 30:
                violations.append(f"{func.__name__} exceeded 30s query time limit: {execution_time}s")
            elif 'visualization' in func.__name__ and execution_time > 5:
                violations.append(f"{func.__name__} exceeded 5s visualization time limit: {execution_time}s")
            elif 'export' in func.__name__ and execution_time > 60:
                violations.append(f"{func.__name__} exceeded 60s export time limit: {execution_time}s")
        except Exception as e:
            violations.append(f"Could not check performance for {func.__name__}: {str(e)}")
        return violations

    def run_all_checks(self) -> Dict[str, List[str]]:
        """Run all compliance checks and return violations."""
        results = {
            'file_length': [],
            'type_hints': [],
            'docstrings': [],
            'test_coverage': [],
            'performance': []
        }
        
        # Check Python files
        for python_file in self.project_root.rglob('*.py'):
            if 'tests' not in str(python_file) and 'venv' not in str(python_file):
                results['file_length'].extend(self.check_file_length(python_file))
                results['type_hints'].extend(self.check_type_hints(python_file))
                results['docstrings'].extend(self.check_docstrings(python_file))
        
        # Run test coverage check
        coverage_percent, coverage_violations = self.check_test_coverage()
        results['test_coverage'].extend(coverage_violations)
        
        return results

def main():
    project_root = Path(os.getcwd())
    checker = RulesChecker(project_root)
    violations = checker.run_all_checks()
    
    # Print results
    print("\n=== Rules Compliance Report ===\n")
    for category, category_violations in violations.items():
        if category_violations:
            print(f"\n{category.upper()} Violations:")
            for violation in category_violations:
                print(f"- {violation}")
    
    # Exit with status code
    has_violations = any(violations.values())
    sys.exit(1 if has_violations else 0)

if __name__ == '__main__':
    main() 