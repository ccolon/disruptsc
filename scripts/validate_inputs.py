#!/usr/bin/env python3
"""
Standalone script to validate input files for DisruptSC model.

Usage:
    python validate_inputs.py <region>
    
Examples:
    python validate_inputs.py Cambodia
    python validate_inputs.py ECA
    python validate_inputs.py Testkistan
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from disruptsc.parameters import Parameters
from disruptsc import paths
from disruptsc.model.input_validation import validate_inputs

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_inputs.py <region>")
        print("Available regions: Cambodia, ECA, Ecuador, Global, Testkistan")
        sys.exit(1)
        
    scope = sys.argv[1]
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print(f"Validating input files for region: {scope}")
    print("=" * 50)
    
    try:
        # Load parameters
        parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, scope)
        
        # Run validation
        is_valid, errors, warnings = validate_inputs(parameters)
        
        # Display results
        if warnings:
            print(f"\n⚠ WARNINGS ({len(warnings)}):")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        
        if errors:
            print(f"\n✗ ERRORS ({len(errors)}):")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
            print(f"\n❌ Validation FAILED - {len(errors)} errors must be fixed before running model")
            sys.exit(1)
        else:
            print(f"\n✅ Validation PASSED - All required files are valid!")
            if warnings:
                print(f"Note: {len(warnings)} warnings found - review for data quality issues")
                
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()