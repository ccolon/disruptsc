"""
Input file validation module for DisruptSC model.

This module provides comprehensive validation of input files to catch errors
before model initialization and provide clear diagnostic information.
"""

import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
import numpy as np


class InputValidationError(Exception):
    """Custom exception for input validation errors."""
    pass

class InputValidator:
    """Validates all input files required for DisruptSC model execution."""
    
    def __init__(self, parameters):
        """Initialize validator with loaded parameters."""
        self.parameters = parameters
        self.scope = parameters.scope
        self.input_folder = Path(parameters.filepaths['sector_table']).parent.parent
        self.errors = []
        self.warnings = []
        
    def validate_all_inputs(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate all required input files.
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        logging.info(f"Starting input validation for scope: {self.scope}")
        
        # Core validation checks
        self._validate_file_existence()
        self._validate_sector_table()
        self._validate_transport_files()
        
        # Mode-specific validation
        if self.parameters.firm_data_type == "mrio":
            self._validate_mrio_inputs()
        elif self.parameters.firm_data_type == "supplier-buyer network":
            self._validate_supplier_buyer_inputs()
        else:
            self.errors.append(f"Unknown firm_data_type: {self.parameters.firm_data_type}. "
                             f"Supported types: 'mrio', 'supplier-buyer network'")
            
        # Parameter consistency checks
        self._validate_parameter_consistency()
        
        # Summary
        is_valid = len(self.errors) == 0
        if is_valid:
            logging.info("✓ All input validation checks passed")
        else:
            logging.error(f"✗ Input validation failed with {len(self.errors)} errors")
            
        return is_valid, self.errors, self.warnings
    
    def _validate_file_existence(self):
        """Check that all required files exist and are readable."""
        required_files = [
            'sector_table',
            'transport_parameters'
        ]
        
        for file_key in required_files:
            filepath = self.parameters.filepaths.get(file_key)
            if filepath and not Path(filepath).exists():
                self.errors.append(f"Required file not found: {filepath}")
        
        # Check transport mode files
        transport_folder = self.input_folder / "Transport"
        if transport_folder.exists():
            for mode in self.parameters.transport_modes:
                edges_file = transport_folder / f"{mode}_edges.geojson"
                if not edges_file.exists():
                    self.errors.append(f"Transport file not found: {edges_file}")
    
    def _validate_sector_table(self):
        """Validate the sector table structure and content."""
        filepath = self.parameters.filepaths.get('sector_table')
        if not filepath or not Path(filepath).exists():
            return
            
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            self.errors.append(f"Cannot read sector_table.csv: {e}")
            return
            
        # Required columns for all modes
        required_cols = ['sector', 'type', 'output', 'final_demand', 'usd_per_ton']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.errors.append(f"sector_table.csv missing required columns: {missing_cols}")
            
        # Mode-specific column requirements
        if self.parameters.firm_data_type == "mrio":
            if 'region_sector' not in df.columns and ('region' not in df.columns or 'sector' not in df.columns):
                self.errors.append("sector_table.csv for MRIO mode must have 'region_sector' column or both 'region' and 'sector' columns")
        
        # Data quality checks
        if 'output' in df.columns:
            if (df['output'] < 0).any():
                self.warnings.append("sector_table.csv contains negative output values")
            if (df['output'] == 0).sum() > len(df) * 0.5:
                self.warnings.append("sector_table.csv has many zero output values - check data quality")
                
        # Sector type validation
        if 'type' in df.columns:
            valid_types = ['agriculture', 'construction', 'mining', 'manufacturing', 'utility', 'transport', 'trade',
                           'service', 'services']
            invalid_types = df[~df['type'].isin(valid_types)]['type'].unique()
            if len(invalid_types) > 0:
                self.warnings.append(f"sector_table.csv contains non-standard sector types: {invalid_types}")
    
    def _validate_transport_files(self):
        """Validate transport network GeoJSON files."""
        transport_folder = self.input_folder / "Transport"
        
        for mode in self.parameters.transport_modes:
            edges_file = transport_folder / f"{mode}_edges.geojson"
            if not edges_file.exists():
                continue
                
            try:
                gdf = gpd.read_file(edges_file)
            except Exception as e:
                self.errors.append(f"Cannot read {edges_file}: {e}")
                continue
                
            # Geometry validation
            if not all(gdf.geometry.geom_type == 'LineString'):
                self.errors.append(f"{edges_file}: All geometries must be LineString")
                
            # Required columns
            if 'km' not in gdf.columns:
                self.warnings.append(f"{edges_file}: Missing 'km' column - distances will be calculated")
                
            # Check for reasonable values
            if 'capacity' in gdf.columns:
                if (gdf['capacity'] < 0).any():
                    self.errors.append(f"{edges_file}: Negative capacity values found")
                    
            if 'km' in gdf.columns:
                if (gdf['km'] <= 0).any():
                    self.errors.append(f"{edges_file}: Non-positive distance values found")
                if (gdf['km'] > 10000).any():
                    self.warnings.append(f"{edges_file}: Very long edges (>10,000 km) found - check units")
    
    def _validate_mrio_inputs(self):
        """Validate inputs specific to MRIO mode."""
        # MRIO table is mandatory - throw error if missing
        mrio_file = self.parameters.filepaths.get('mrio')
        if not mrio_file:
            self.errors.append("MRIO mode requires 'mrio' filepath to be specified in parameters")
        elif not Path(mrio_file).exists():
            self.errors.append(f"Required MRIO table not found: {mrio_file}")
        else:
            self._validate_mrio_table(mrio_file)
            
        # Spatial files are mandatory - throw error if missing
        self._validate_spatial_files()
    
    def _validate_mrio_table(self, filepath):
        """Validate the multi-regional input-output table structure."""
        try:
            df = pd.read_csv(filepath, index_col=[0, 1], header=[0, 1])
        except Exception as e:
            self.errors.append(f"Cannot read MRIO table as multi-index: {e}")
            return

        # Check for completely empty data (critical error)
        if df.empty or df.isna().all().all():
            self.errors.append("MRIO table is empty or contains only NaN values")
            return
            
        # Check for negative values in intermediate flows (error, not warning)
        intermediate_cols = [col for col in df.columns if not any(
            keyword in str(col).lower() for keyword in ['final', 'export', 'capital', 'government']
        )]
        intermediate_rows = [row for row in df.index if not any(
            keyword in str(row).lower() for keyword in ['value', 'va', 'import', 'tax']
        )]
        if (len(intermediate_cols) == 0) or (len(intermediate_rows) == 0):
            self.errors.append("No matrix of intermediate flows detected in MRIO table")

        intermediate_df = df.loc[intermediate_rows, intermediate_cols]
        if intermediate_df.shape[0] != intermediate_df.shape[1]:
            self.errors.append(f"The intermediary part of the MRIO table must be square: "
                               f"{intermediate_df.shape[0]} rows vs {intermediate_df.shape[1]} columns")
        if (intermediate_df < 0).any().any():
            self.errors.append("MRIO table contains negative intermediate flows")
                
        # Check for extreme imbalance (critical error)
        try:
            row_sums = df.sum(axis=1)[intermediate_rows]
            col_sums = df.sum(axis=0)[intermediate_cols]
            if not np.allclose(row_sums, col_sums, rtol=0.5):
                self.errors.append("MRIO table is severely unbalanced (row sums ≠ column sums)")
            elif not np.allclose(row_sums, col_sums, rtol=0.1):
                self.warnings.append("MRIO table appears unbalanced (row sums ≠ column sums)")
        except Exception as e:
            self.errors.append(f"Cannot compute MRIO table balance: {e}")
    
    def _validate_spatial_files(self):
        """Validate new spatial file structure."""
        spatial_files = {
            'households_spatial': 'households.geojson',
            'countries_spatial': 'countries.geojson', 
            'firms_spatial': 'firms.geojson'
        }
        
        for file_key, filename in spatial_files.items():
            filepath = self.parameters.filepaths.get(file_key)
            if not filepath:
                self.errors.append(f"MRIO mode requires '{file_key}' filepath to be specified in parameters")
                continue
                
            if not Path(filepath).exists():
                self.errors.append(f"Required spatial file not found: {filepath}")
                continue
                
            self._validate_spatial_file(filepath, filename)
        
        # Warn about deprecated files
        deprecated_files = ['region_table']
        for deprecated_key in deprecated_files:
            if self.parameters.filepaths.get(deprecated_key):
                self.warnings.append(f"Deprecated parameter '{deprecated_key}' found. "
                                   f"Use 'households_spatial' and 'countries_spatial' instead.")
        
        # Check for deprecated Disag folder
        spatial_folder = Path(self.parameters.filepaths.get('households_spatial', '')).parent
        disag_folder = spatial_folder / "Disag"
        if disag_folder.exists():
            self.warnings.append("Deprecated Disag/ folder found. Use firms.geojson instead.")
    
    def _validate_spatial_file(self, filepath, filename):
        """Validate individual spatial file structure."""
        try:
            gdf = gpd.read_file(filepath)
        except Exception as e:
            self.errors.append(f"Cannot read {filename}: {e}")
            return
            
        # Check for empty data (critical error)
        if gdf.empty:
            self.errors.append(f"{filename} is empty")
            return
            
        # Check for required identifier column (critical error)
        if 'region' not in gdf.columns:
            self.errors.append(f"{filename} must have a 'region' column")
            
        # Check geometry type (critical error)
        if not all(gdf.geometry.geom_type == 'Point'):
            self.errors.append(f"{filename}: All geometries must be Points")
            
        # Check for missing geometries (critical error)
        if gdf.geometry.isna().any():
            self.errors.append(f"{filename} contains missing geometries")
            
        # Check for duplicate region identifiers (critical error)
        id_col = 'admin_code' if 'admin_code' in gdf.columns else 'region'
        if id_col in gdf.columns:
            if gdf[id_col].duplicated().any():
                self.errors.append(f"{filename} contains duplicate {id_col} values")
            if gdf[id_col].isna().any():
                self.errors.append(f"{filename} contains missing {id_col} values")
    
    def _validate_supplier_buyer_inputs(self):
        """Validate inputs specific to supplier-buyer network mode."""
        # Transaction table is required for supplier-buyer network mode
        transaction_file = self.parameters.filepaths.get('transaction_table')
        if not transaction_file:
            self.errors.append("Supplier-buyer network mode requires 'transaction_table' filepath to be specified")
        elif not Path(transaction_file).exists():
            self.errors.append(f"Required transaction table not found: {transaction_file}")
        else:
            self._validate_transaction_table(transaction_file)
    
    def _validate_transaction_table(self, filepath):
        """Validate transaction table structure."""
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            self.errors.append(f"Cannot read transaction_table.csv: {e}")
            return
            
        # Check required columns for supplier-buyer relationships
        required_cols = ['supplier_id', 'buyer_id', 'value', 'sector']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.errors.append(f"transaction_table.csv missing required columns: {missing_cols}")
            
        # Check for negative transaction values
        if 'value' in df.columns and (df['value'] < 0).any():
            self.errors.append("transaction_table.csv contains negative transaction values")
    
    def _validate_parameter_consistency(self):
        """Validate parameter consistency and reasonable values."""
        # Monetary units consistency
        valid_units = ['USD', 'kUSD', 'mUSD']
        if self.parameters.monetary_units_in_model not in valid_units:
            self.warnings.append(f"Unusual monetary unit in model: {self.parameters.monetary_units_in_model}")
            
        if self.parameters.monetary_units_in_data not in valid_units:
            self.warnings.append(f"Unusual monetary unit in data: {self.parameters.monetary_units_in_data}")
            
        # Cutoff values
        if self.parameters.io_cutoff < 0 or self.parameters.io_cutoff > 1:
            self.warnings.append(f"IO cutoff should typically be between 0 and 1, got: {self.parameters.io_cutoff}")
            
        # Time parameters
        if self.parameters.t_final <= 0:
            self.errors.append(f"t_final must be positive, got: {self.parameters.t_final}")
            
        # Transport modes
        valid_transport_modes = ['roads', 'railways', 'maritime', 'waterways', 'airways', 'pipelines', 'multimodal']
        invalid_modes = [mode for mode in self.parameters.transport_modes if mode not in valid_transport_modes]
        if invalid_modes:
            self.warnings.append(f"Non-standard transport modes specified: {invalid_modes}")


def validate_inputs(parameters) -> Tuple[bool, List[str], List[str]]:
    """
    Convenience function to validate all inputs for a given parameter set.
    
    Args:
        parameters: Loaded Parameters object
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = InputValidator(parameters)
    return validator.validate_all_inputs()


def main():
    """Command-line interface for input validation."""
    import argparse
    import sys
    from disruptsc.parameters import Parameters
    from disruptsc import paths
    
    parser = argparse.ArgumentParser(description="Validate DisruptSC input files")
    parser.add_argument("scope", help="Region/scope to validate")
    parser.add_argument("--version", action="version", version=f"DisruptSC {__import__('disruptsc').__version__}")
    args = parser.parse_args()
    
    try:
        parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, args.scope)
        is_valid, errors, warnings = validate_inputs(parameters)
        
        # Print results
        if warnings:
            print("WARNINGS:")
            for warning in warnings:
                print(f"  ⚠ {warning}")
            print()
            
        if errors:
            print("ERRORS:")
            for error in errors:
                print(f"  ✗ {error}")
            print()
            print(f"Validation failed with {len(errors)} errors")
            sys.exit(1)
        else:
            print("✓ All validation checks passed!")
            
    except Exception as e:
        print(f"Validation failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()