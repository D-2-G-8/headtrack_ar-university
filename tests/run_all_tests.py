#!/usr/bin/env python3
"""
Run all tests for headtrack_ar package.

This script discovers and runs all test modules in the tests directory.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def discover_and_run_tests():
    """Discover and run all tests."""
    # Discover tests in the tests directory
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(Path(__file__).parent),
        pattern='test_*.py',
        top_level_dir=str(project_root)
    )
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


def run_specific_test_suite(suite_name):
    """Run a specific test suite.
    
    Args:
        suite_name: Name of test suite (e.g., 'basic', 'integration', 'performance')
    """
    loader = unittest.TestLoader()
    
    test_modules = {
        'basic': 'tests.test_basic',
        'detector': 'tests.test_detector',
        'video_source': 'tests.test_video_source',
        'integration': 'tests.test_integration',
        'performance': 'tests.test_performance',
        'conditions': 'tests.test_conditions',
        'error_handling': 'tests.test_error_handling',
    }
    
    if suite_name not in test_modules:
        print(f"Unknown test suite: {suite_name}")
        print(f"Available suites: {', '.join(test_modules.keys())}")
        return 1
    
    module_name = test_modules[suite_name]
    suite = loader.loadTestsFromName(module_name)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific test suite
        suite_name = sys.argv[1]
        exit_code = run_specific_test_suite(suite_name)
    else:
        # Run all tests
        exit_code = discover_and_run_tests()
    
    sys.exit(exit_code)
