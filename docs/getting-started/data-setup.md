# Data Setup

DisruptSC requires input data to run simulations. This guide explains how to set up your data sources.

## Repository Separation

The model code and data are maintained in separate repositories:

- **disrupt-sc** (public) - Model code, configuration, documentation
- **disrupt-sc-data** (private) - Input data files

This separation allows for:
- Public sharing of the model code
- Secure handling of sensitive economic data
- Flexible data source configuration

## Data Setup Options

Choose one of the following methods to provide input data:

### Option 1: Git Submodule (Recommended)

If you have access to the private data repository:

```bash
# Add data repository as submodule
git submodule add <disrupt-sc-data-repository-url> data
git submodule update --init

# Update data to latest version
git submodule update --remote data
```

**Advantages:**
- Version-controlled data
- Automatic updates with git
- Collaborative data management

### Option 2: Environment Variable (Flexible)

Point to a custom data directory location:

```bash
# Set environment variable (Linux/macOS)
export DISRUPT_SC_DATA_PATH=/path/to/your/data/folder

# Set environment variable (Windows)
set DISRUPT_SC_DATA_PATH=C:\path\to\your\data\folder

# Make permanent (add to ~/.bashrc, ~/.zshrc, etc.)
echo 'export DISRUPT_SC_DATA_PATH=/path/to/data' >> ~/.bashrc
```

**Advantages:**
- Flexible data location
- Multiple data sources
- Development-friendly

### Option 3: Local Input Folder (Simple)

Create a local `input/` folder:

```bash
# Create input directory
mkdir input

# Copy your data files following the structure below
# input/Cambodia/Economic/mrio.csv
# input/Cambodia/Transport/roads_edges.geojson
# etc.
```

**Advantages:**
- Simple setup
- No external dependencies
- Good for testing

## Data Path Priority

DisruptSC automatically detects data location in this order:

1. **`DISRUPT_SC_DATA_PATH`** environment variable (highest priority)
2. **`data/`** folder (git submodule)
3. **`input/`** folder (local fallback)

## Required Data Structure

Input data must be organized by scope (region):

```
data/{scope}/               # e.g., data/Cambodia/
├── Economic/               # Economic data
│   ├── mrio.csv           # Multi-regional input-output table
│   └── sector_table.csv   # Sector definitions and parameters
├── Transport/              # Infrastructure networks
│   ├── roads_edges.geojson          # Road network (LineString)
│   ├── maritime_edges.geojson       # Maritime routes (LineString) 
│   ├── railways_edges.geojson       # Railway network (LineString)
│   ├── airways_edges.geojson        # Air routes (LineString)
│   ├── waterways_edges.geojson      # Waterway network (LineString)
│   ├── pipelines_edges.geojson      # Pipeline network (LineString)
│   └── multimodal_edges.geojson     # Multimodal connections
└── Spatial/                # Geographic disaggregation
    ├── households.geojson           # Household locations (Point)
    ├── countries.geojson            # Country entry points (Point)
    └── firms.geojson                # Firm spatial distribution (Point)
```

## Scope Configuration

Each scope requires:

1. **Data folder**: `data/{scope}/` or `input/{scope}/`
2. **Parameter file**: `parameter/user_defined_{scope}.yaml`

For example, to set up Cambodia:
- Data: `data/Cambodia/`
- Parameters: `parameter/user_defined_Cambodia.yaml`

## File Requirements

### Essential Files (Always Required)
- `Economic/mrio.csv` - Input-output table
- `Economic/sector_table.csv` - Sector definitions
- `Transport/roads_edges.geojson` - Road network
- `Spatial/households.geojson` - Household locations

### Transport Networks
At minimum, roads are required. Additional transport modes are optional:
- Maritime (international trade)
- Railways (freight transport)
- Airways (high-value goods)
- Waterways (bulk transport)
- Pipelines (energy/chemicals)

### Data Modes
Different data requirements based on mode:

!!! info "MRIO Mode (Default)"
    
    **Required:**
    - `Economic/mrio.csv`
    - `Economic/sector_table.csv`
    - `Spatial/*.geojson` files
    
    **Generated:** Firms, households, countries from MRIO data

!!! tip "Supplier-Buyer Network Mode"
    
    **Additional Requirements:**
    - `Economic/firm_table.csv`
    - `Economic/location_table.csv`
    - `Economic/transaction_table.csv`
    
    **Use case:** When you have detailed firm-level data

## Verification

After setting up your data, verify the configuration:

```bash
# Check data path detection
python -c "from disruptsc.paths import get_data_path; print(get_data_path('Cambodia'))"

# Validate input files
python validate_inputs.py Cambodia

# Test basic model initialization
python disruptsc/main.py Cambodia --help
```

## Troubleshooting

### Data Path Issues

??? failure "Data path not found"
    
    ```bash
    # Check if environment variable is set
    echo $DISRUPT_SC_DATA_PATH
    
    # Verify folder exists
    ls -la data/Cambodia/
    ls -la input/Cambodia/
    
    # Check folder permissions
    ls -la data/
    ```

??? failure "Git submodule problems"
    
    ```bash
    # Initialize submodules
    git submodule update --init --recursive
    
    # Reset submodule
    git submodule deinit data
    git rm data
    git submodule add <url> data
    ```

### File Format Issues

??? failure "Invalid file formats"
    
    - Ensure CSV files use UTF-8 encoding
    - GeoJSON files must have valid geometry
    - LineString required for transport edges
    - Point geometry required for spatial locations

??? failure "Missing required columns"
    
    Run the input validator for detailed error messages:
    ```bash
    python validate_inputs.py Cambodia
    ```

## What's Next?

After setting up your data:

1. **[Validate inputs](validation.md)** - Check data quality
2. **[Run quick start](quick-start.md)** - Test your setup
3. **[Configure parameters](../user-guide/parameters.md)** - Customize simulation settings