#!/usr/bin/env python3
"""
Test suite for input file validation.

This script tests the validation logic with various scenarios including
intentionally malformed files to ensure proper error detection.
"""

import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path
import tempfile
import shutil
from shapely.geometry import Point, LineString

from disruptsc.model.input_validation import InputValidator, validate_inputs
from disruptsc.parameters import Parameters


class TestInputValidation:
    """Test cases for input validation functionality."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.create_minimal_valid_files()
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def create_minimal_valid_files(self):
        """Create minimal valid input files for testing."""
        # Create basic directory structure
        (self.test_dir / "parameter").mkdir()
        (self.test_dir / "input" / "TestRegion" / "Network").mkdir(parents=True)
        (self.test_dir / "input" / "TestRegion" / "Transport").mkdir(parents=True)
        
        # Create minimal parameters
        default_params = {
            'scope': 'TestRegion',
            'firm_data_type': 'mrio',
            'transport_modes': ['roads'],
            'monetary_units_in_model': 'USD',
            'monetary_units_in_data': 'USD',
            'time_resolution': 'week',
            'io_cutoff': 0.01,
            't_final': 52,
            'filepaths': {
                'sector_table': 'Network/sector_table.csv',
                'mrio': 'Network/mrio.csv',
                'region_table': 'Network/region_table.geojson',
                'transport_parameters': 'Transport/transport_parameters.yaml'
            }
        }
        
        with open(self.test_dir / "parameter" / "default.yaml", 'w') as f:
            import yaml
            yaml.dump(default_params, f)
        
        # Create valid sector table
        sector_df = pd.DataFrame({
            'sector': ['AGR', 'MAN', 'SER'],
            'region_sector': ['REG1_AGR', 'REG1_MAN', 'REG1_SER'],
            'region': ['REG1', 'REG1', 'REG1'],
            'type': ['agriculture', 'manufacturing', 'service'],
            'output': [1000, 2000, 1500],
            'final_demand': [800, 1200, 1000],
            'usd_per_ton': [500, 2000, 0]
        })
        sector_df.to_csv(self.test_dir / "input" / "TestRegion" / "Network" / "sector_table.csv", index=False)
        
        # Create valid MRIO table
        mrio_data = pd.DataFrame(
            [[100, 50, 20, 200], [30, 150, 40, 300], [10, 20, 100, 150], [0, 0, 0, 0]],
            index=pd.MultiIndex.from_tuples([('REG1', 'AGR'), ('REG1', 'MAN'), ('REG1', 'SER'), ('ROW', 'imports')]),
            columns=pd.MultiIndex.from_tuples([('REG1', 'AGR'), ('REG1', 'MAN'), ('REG1', 'SER'), ('REG1', 'final_demand')])
        )
        mrio_data.to_csv(self.test_dir / "input" / "TestRegion" / "Network" / "mrio.csv")
        
        # Create valid region table
        region_gdf = gpd.GeoDataFrame({
            'admin_code': ['REG1'],
            'region': ['REG1'],
            'population': [100000],
            'geometry': [Point(0, 0)]
        })
        region_gdf.to_file(self.test_dir / "input" / "TestRegion" / "Network" / "region_table.geojson", driver="GeoJSON")
        
        # Create valid transport files
        transport_gdf = gpd.GeoDataFrame({
            'id': [1, 2, 3],
            'km': [10.5, 15.2, 8.7],
            'capacity': [1000, 1500, 800],
            'surface': ['paved', 'paved', 'unpaved'],
            'geometry': [
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]), 
                LineString([(0, 0), (0, 1)])
            ]
        })
        transport_gdf.to_file(self.test_dir / "input" / "TestRegion" / "Transport" / "roads_edges.geojson", driver="GeoJSON")
        
        # Create transport parameters
        transport_params = {
            'cost_per_km': {'roads': 0.1},
            'speed_kmh': {'roads': 50}
        }
        with open(self.test_dir / "input" / "TestRegion" / "Transport" / "transport_parameters.yaml", 'w') as f:
            import yaml
            yaml.dump(transport_params, f)
    
    def test_valid_mrio_setup(self):
        """Test that valid MRIO setup passes validation."""
        # Mock paths module
        import disruptsc.paths as paths
        original_input = paths.INPUT_FOLDER
        original_param = paths.PARAMETER_FOLDER
        
        try:
            paths.INPUT_FOLDER = self.test_dir / "input"
            paths.PARAMETER_FOLDER = self.test_dir / "parameter"
            
            parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, "TestRegion")
            is_valid, errors, warnings = validate_inputs(parameters)
            
            assert is_valid, f"Validation should pass, but got errors: {errors}"
            
        finally:
            paths.INPUT_FOLDER = original_input
            paths.PARAMETER_FOLDER = original_param
    
    def test_missing_mrio_table(self):
        """Test that missing MRIO table throws error."""
        # Remove MRIO file
        (self.test_dir / "input" / "TestRegion" / "Network" / "mrio.csv").unlink()
        
        import disruptsc.paths as paths
        original_input = paths.INPUT_FOLDER
        original_param = paths.PARAMETER_FOLDER
        
        try:
            paths.INPUT_FOLDER = self.test_dir / "input"
            paths.PARAMETER_FOLDER = self.test_dir / "parameter"
            
            parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, "TestRegion")
            is_valid, errors, warnings = validate_inputs(parameters)
            
            assert not is_valid
            assert any("Required MRIO table not found" in error for error in errors)
            
        finally:
            paths.INPUT_FOLDER = original_input
            paths.PARAMETER_FOLDER = original_param
    
    def test_missing_region_table(self):
        """Test that missing region table throws error."""
        # Remove region table file
        (self.test_dir / "input" / "TestRegion" / "Network" / "region_table.geojson").unlink()
        
        import disruptsc.paths as paths
        original_input = paths.INPUT_FOLDER
        original_param = paths.PARAMETER_FOLDER
        
        try:
            paths.INPUT_FOLDER = self.test_dir / "input"
            paths.PARAMETER_FOLDER = self.test_dir / "parameter"
            
            parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, "TestRegion")
            is_valid, errors, warnings = validate_inputs(parameters)
            
            assert not is_valid
            assert any("Required region table not found" in error for error in errors)
            
        finally:
            paths.INPUT_FOLDER = original_input
            paths.PARAMETER_FOLDER = original_param
    
    def test_missing_required_columns(self):
        """Test detection of missing required columns."""
        # Remove required column from sector table
        sector_df = pd.read_csv(self.test_dir / "input" / "TestRegion" / "Network" / "sector_table.csv")
        sector_df = sector_df.drop(columns=['output'])
        sector_df.to_csv(self.test_dir / "input" / "TestRegion" / "Network" / "sector_table.csv", index=False)
        
        import disruptsc.paths as paths
        original_input = paths.INPUT_FOLDER
        original_param = paths.PARAMETER_FOLDER
        
        try:
            paths.INPUT_FOLDER = self.test_dir / "input"
            paths.PARAMETER_FOLDER = self.test_dir / "parameter"
            
            parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, "TestRegion")
            is_valid, errors, warnings = validate_inputs(parameters)
            
            assert not is_valid
            assert any("missing required columns" in error for error in errors)
            
        finally:
            paths.INPUT_FOLDER = original_input
            paths.PARAMETER_FOLDER = original_param
    
    def test_invalid_transport_geometry(self):
        """Test detection of invalid transport geometries."""
        # Create transport file with invalid geometry (Point instead of LineString)
        transport_gdf = gpd.GeoDataFrame({
            'id': [1],
            'km': [10.5],
            'geometry': [Point(0, 0)]  # Invalid: should be LineString
        })
        transport_gdf.to_file(self.test_dir / "input" / "TestRegion" / "Transport" / "roads_edges.geojson", driver="GeoJSON")
        
        import disruptsc.paths as paths
        original_input = paths.INPUT_FOLDER
        original_param = paths.PARAMETER_FOLDER
        
        try:
            paths.INPUT_FOLDER = self.test_dir / "input"
            paths.PARAMETER_FOLDER = self.test_dir / "parameter"
            
            parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, "TestRegion")
            is_valid, errors, warnings = validate_inputs(parameters)
            
            assert not is_valid
            assert any("LineString" in error for error in errors)
            
        finally:
            paths.INPUT_FOLDER = original_input
            paths.PARAMETER_FOLDER = original_param
    
    def test_negative_values_detection(self):
        """Test detection of negative values in economic data."""
        # Add negative values to sector table
        sector_df = pd.read_csv(self.test_dir / "input" / "TestRegion" / "Network" / "sector_table.csv")
        sector_df.loc[0, 'output'] = -1000  # Invalid negative output
        sector_df.to_csv(self.test_dir / "input" / "TestRegion" / "Network" / "sector_table.csv", index=False)
        
        import disruptsc.paths as paths
        original_input = paths.INPUT_FOLDER
        original_param = paths.PARAMETER_FOLDER
        
        try:
            paths.INPUT_FOLDER = self.test_dir / "input"
            paths.PARAMETER_FOLDER = self.test_dir / "parameter"
            
            parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, "TestRegion")
            is_valid, errors, warnings = validate_inputs(parameters)
            
            # Should pass validation but generate warning
            assert any("negative output values" in warning for warning in warnings)
            
        finally:
            paths.INPUT_FOLDER = original_input
            paths.PARAMETER_FOLDER = original_param


def run_validation_tests():
    """Run all validation tests manually (for environments without pytest)."""
    test_class = TestInputValidation()
    
    test_methods = [
        'test_valid_mrio_setup',
        'test_missing_mrio_table',
        'test_missing_region_table',
        'test_missing_required_columns', 
        'test_invalid_transport_geometry',
        'test_negative_values_detection'
    ]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            print(f"Running {method_name}...")
            test_class.setup_method()
            method = getattr(test_class, method_name)
            method()
            test_class.teardown_method()
            print(f"  ✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            try:
                test_class.teardown_method()
            except:
                pass
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    # Try to use pytest if available, otherwise run manually
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running tests manually...")
        success = run_validation_tests()
        exit(0 if success else 1)