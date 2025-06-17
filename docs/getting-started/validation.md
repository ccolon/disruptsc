# Input Validation

Before running simulations, validate your input files to catch errors early and ensure data quality.

## Why Validate?

Input validation helps you:

- **Catch errors early** before long simulation runs
- **Ensure data consistency** across files
- **Verify file formats** and required columns
- **Check data quality** and value ranges
- **Get actionable feedback** for fixing issues

## Quick Validation

Validate all input files for a scope:

```bash
python validate_inputs.py <scope>
```

**Examples:**
```bash
python validate_inputs.py Cambodia
python validate_inputs.py Ecuador
python validate_inputs.py Global
```

## What Gets Validated

### File Existence and Access
- All required files are present
- Files are readable
- Directory structure is correct

### Economic Data
- **MRIO table** structure and balance
- **Sector table** required columns and data types
- **Parameter consistency** between files
- **Monetary units** alignment

### Transport Networks
- **GeoJSON format** validity
- **LineString geometry** for edges
- **Coordinate systems** consistency
- **Network connectivity** basic checks

### Spatial Data
- **Point geometry** for locations
- **Required attributes** present
- **Region matching** with economic data
- **Spatial reference** consistency

### Data Ranges and Quality
- **No negative values** where inappropriate
- **Reasonable value ranges** for economic variables
- **Missing data** identification
- **Outlier detection** for key variables

## Validation Output

### Success Message
When all validations pass:
```
✅ Validation successful for Cambodia
All required files found and validated
Ready to run simulations
```

### Error Messages
When validation fails:
```
❌ Validation failed for Cambodia

Errors found:
- Missing file: data/Cambodia/Economic/mrio.csv
- Invalid geometry in data/Cambodia/Transport/roads_edges.geojson
- MRIO table is not balanced (rows != columns)

Warnings:
- Deprecated file found: data/Cambodia/legacy_data.csv
- Large file detected: data/Cambodia/Transport/roads_edges.geojson (>50MB)
```

### Detailed Output
For verbose information:
```bash
python validate_inputs.py Cambodia --verbose
```

## Common Issues and Solutions

### File Errors

??? failure "Missing required files"
    
    **Error:** `Missing file: data/Cambodia/Economic/mrio.csv`
    
    **Solutions:**
    - Check data path configuration
    - Verify file names match exactly
    - Ensure proper scope folder structure
    
    ```bash
    # Check data structure
    ls -la data/Cambodia/Economic/
    
    # Verify data path
    python -c "from disruptsc.paths import get_data_path; print(get_data_path('Cambodia'))"
    ```

??? failure "File access permissions"
    
    **Error:** `Cannot read file: permission denied`
    
    **Solutions:**
    ```bash
    # Fix file permissions
    chmod 644 data/Cambodia/Economic/mrio.csv
    
    # Fix directory permissions
    chmod 755 data/Cambodia/
    ```

### Economic Data Errors

??? failure "MRIO table not balanced"
    
    **Error:** `MRIO table is not balanced: sum(inputs) != sum(outputs)`
    
    **Solutions:**
    - Check input-output table calculation
    - Verify monetary units consistency
    - Review aggregation methods
    - Ensure final demand is included

??? failure "Missing required columns"
    
    **Error:** `Missing column 'sector' in sector_table.csv`
    
    **Required columns in sector_table.csv:**
    - `sector` - Region_sector identifier
    - `type` - Sector type (agriculture, manufacturing, etc.)
    - `output` - Total yearly output
    - `final_demand` - Total yearly final demand
    - `usd_per_ton` - USD value per ton
    - `share_exporting_firms` - Export participation rate

### Transport Network Errors

??? failure "Invalid geometry"
    
    **Error:** `Invalid geometry in roads_edges.geojson`
    
    **Solutions:**
    - Ensure LineString geometry for transport edges
    - Check for invalid coordinates (NaN, infinity)
    - Verify coordinate reference system
    - Use QGIS or similar tools to repair geometry

??? failure "Network connectivity issues"
    
    **Error:** `Disconnected transport network components`
    
    **Solutions:**
    - Check for isolated network segments
    - Ensure proper node connectivity
    - Review multimodal connections
    - Add connecting edges where needed

### Spatial Data Errors

??? failure "Mismatched regions"
    
    **Error:** `Region 'REG01' in households.geojson not found in MRIO`
    
    **Solutions:**
    - Check region naming consistency
    - Verify MRIO region definitions
    - Update spatial data region attributes
    - Check for typos in region codes

## Integration with Model

The model automatically runs basic validation at startup:

```python
from disruptsc.model.input_validation import validate_inputs
from disruptsc.parameters import Parameters

# Load parameters
parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, scope)

# Validate inputs
is_valid, errors, warnings = validate_inputs(parameters)

if not is_valid:
    print("Validation errors:", errors)
    exit(1)
```

## Advanced Validation

### Custom Validation Rules

For project-specific validation:

```python
from disruptsc.model.input_validation import BaseValidator

class CustomValidator(BaseValidator):
    def validate_custom_data(self):
        # Your custom validation logic
        pass
```

### Batch Validation

Validate multiple scopes:

```bash
for scope in Cambodia Ecuador Global; do
    echo "Validating $scope..."
    python validate_inputs.py $scope
done
```

## Performance Tips

For large datasets:

- Use `--quick` flag for basic validation only
- Validate subsets during development
- Cache validation results for repeated checks
- Run full validation before production runs

## What's Next?

After successful validation:

1. **[Run quick start](quick-start.md)** - Test your first simulation
2. **[Configure parameters](../user-guide/parameters.md)** - Customize settings
3. **[Understand data modes](../user-guide/data-modes.md)** - Choose appropriate mode