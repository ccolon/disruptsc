# Disaggregating IO Mode - Archive Documentation

**Date Archived:** December 6, 2024  
**Version:** 1.0.8  
**Reason for Removal:** Simplifying codebase to focus on MRIO and supplier-buyer network modes

## Overview

The "disaggregating IO" mode was one of three firm data types supported by DisruptSC:
1. **"mrio"** - Uses multi-regional input-output tables directly
2. **"supplier-buyer network"** - Uses explicit supplier-buyer relationship data  
3. **"disaggregating IO"** - Uses national IO tables disaggregated using regional economic data ⚠️ REMOVED

## How Disaggregating IO Mode Worked

### Concept
- Started with national input-output (IO) tables and technical coefficients
- Used regional economic data to spatially disaggregate firms across regions
- Created firms based on regional supply data for each sector
- Required extensive regional economic datasets

### Data Requirements
```
input/{region}/
├── National/           # Core IO data
│   ├── sector_table.csv       # Sector definitions with supply_data column
│   ├── tech_coef.csv          # Technical coefficient matrix
│   └── inventory_targets.csv  # Inventory duration targets
├── Subnational/        # Regional disaggregation
│   └── region_data.geojson    # Regional economic data with sector-specific columns
└── Trade/              # Trade flows
    ├── import_table.csv
    ├── export_table.csv
    └── transit_matrix.csv
```

### Key Files That Used This Mode
- **Default configuration**: `parameter/default.yaml` (was the default mode)
- **Ecuador**: `parameter/user_defined_Ecuador.yaml` (explicitly configured)
- **Data example**: Ecuador region had full disaggregating IO setup

## Archived Code Components

### 1. Main Firm Builder Function
**File**: `disruptsc/model/firm_builder_functions.py`  
**Function**: `define_firms_from_local_economic_data()`
- **Purpose**: Main entry point for creating firms from regional economic data
- **Logic**: 
  1. Load regional economic data and sector table
  2. For each sector, find regions above supply threshold
  3. Create firms in those regions
  4. Assign to nearest transport nodes
  5. Calculate firm importance scores

### 2. Input Validation
**File**: `disruptsc/model/input_validation.py`  
**Functions**: 
- `_validate_disaggregating_io_inputs()` - Main validation entry point
- `_validate_region_data()` - Validates regional economic data structure

### 3. Household Builder Integration  
**File**: `disruptsc/model/household_builder_functions.py`
- Functions that consumed `filepath_region_data` parameter
- Regional population-based household placement logic

### 4. Model Integration Points
**File**: `disruptsc/model/model.py`
- Conditional logic: `if self.parameters.firm_data_type == "disaggregating IO"`
- Firm creation pathway using regional economic data
- Household creation with regional data integration

## Technical Details

### Sector Table Requirements
```csv
sector,supply_data,sector_type,cutoff
agriculture,agricultural_output,primary,1000000
manufacturing,industrial_output,secondary,5000000
services,service_output,tertiary,2000000
```

### Regional Data Requirements  
```geojson
{
  "type": "Feature", 
  "properties": {
    "region": "Region_001",
    "admin_code": "12345", 
    "population": 50000,
    "pop_density": 120.5,
    "agricultural_output": 1500000,    # Sector-specific columns
    "industrial_output": 8000000,      # Matched to sector_table.supply_data
    "service_output": 3000000
  },
  "geometry": {"type": "Point", "coordinates": [lon, lat]}
}
```

### Algorithm Flow
1. **Sector Processing**: For each sector in sector_table
2. **Regional Filtering**: Find regions where `supply_data > cutoff`  
3. **Firm Placement**: Create firms in qualifying regions
4. **Transport Mapping**: Assign firms to nearest transport network nodes
5. **Size Calculation**: Compute firm importance relative to sector total
6. **Aggregation**: Combine firms at same transport nodes

## Rationale for Removal

### Issues with Disaggregating IO Mode
1. **Data Intensive**: Required extensive regional economic datasets that are hard to obtain
2. **Complexity**: Added significant conditional logic throughout codebase  
3. **Maintenance Burden**: Three different firm creation pathways to maintain
4. **Limited Usage**: Only Ecuador was actively using this mode
5. **Bug Source**: Caused confusion in parameter validation (Cambodia case)

### Benefits of MRIO as Default
1. **Standardization**: Single, well-established methodology
2. **Data Availability**: MRIO tables more commonly available
3. **Simplicity**: Cleaner codebase with fewer conditional branches
4. **Performance**: More efficient without regional disaggregation overhead

## Migration Path

### For Users of Disaggregating IO Mode
1. **Option 1**: Convert to MRIO format
   - Aggregate regional data back to country/regional level
   - Create proper multi-level MRIO table structure
   
2. **Option 2**: Use supplier-buyer network mode
   - If explicit firm relationships are available
   - More detailed than disaggregating IO approach

### Ecuador Migration Example
Ecuador was the main user of disaggregating IO mode. Migration options:
1. Convert Ecuador's regional data to MRIO format
2. Use Ecuador's 59-sector MRIO table (already available)
3. Switch to `firm_data_type: "mrio"` in parameters

## Code Archive Location

All removed code is preserved in:
- **Methods**: `disruptsc/legacy/disaggregating_io_methods.py`
- **Parameters**: `parameter/legacy/`
- **Documentation**: This file (`DISAGGREGATING_IO_ARCHIVE.md`)

## Historical Context

The disaggregating IO mode was created to handle cases where:
- National IO tables were available but lacked spatial detail
- Regional economic census data could provide spatial distribution
- MRIO tables were not available or incomplete

With improved MRIO data availability and standardization, this approach became less necessary while adding significant complexity to the codebase.

---

**Note**: This archive preserves the institutional knowledge and technical approach for future reference. The code remains functional as of version 1.0.8 but is no longer supported in future versions.