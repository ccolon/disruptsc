# Installation

This guide will help you install DisruptSC and set up the required environment.

## System Requirements

- **Python**: 3.10 or 3.11 (required)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large models)
- **Storage**: 5GB+ free space for data and outputs

## Step 1: Clone the Repository

The DisruptSC project uses Git submodules to manage data separately from the main codebase. Choose one of the following methods:

### Method A: Clone with Submodules (Recommended)

```bash
# Clone repository and initialize submodules in one step
git clone --recurse-submodules https://github.com/worldbank/disrupt-sc.git
cd disrupt-sc
```

### Method B: Clone then Initialize Submodules

```bash
# Clone repository
git clone https://github.com/ccolon/disrupt-sc.git
cd disrupt-sc

# Initialize and update data submodule
git submodule update --init --recursive
```

!!! info "About Data Submodules"
    DisruptSC uses a separate private repository for input data to keep the main codebase lightweight. The data is automatically linked as a Git submodule in the `data/` folder.

## Step 2: Environment Setup

We recommend using Conda for environment management to ensure reproducible installations.

### Option A: Conda Environment (Recommended)

```bash
# Create environment from file
conda env create -f dsc-environment.yml

# Activate environment
conda activate dsc

# Install package in development mode
pip install -e .
```

### Option B: Pip Installation

If you prefer using pip or don't have Conda installed:

```bash
# Create virtual environment
python -m venv dsc-env

# Activate environment
# On Windows:
dsc-env\Scripts\activate
# On macOS/Linux:
source dsc-env/bin/activate

# Install dependencies
pip install -e .
```

## Step 3: Verify Installation

Test that DisruptSC is correctly installed:

```bash
# Check version
python -c "import disruptsc; print(disruptsc.__version__)"

# Test CLI
python disruptsc/main.py --version

# Run input validation (requires data setup)
python validate_inputs.py --help
```

## Dependencies

DisruptSC relies on several key packages:

### Core Dependencies
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **geopandas** - Geospatial data processing
- **networkx** - Graph/network analysis
- **scipy** - Scientific computing
- **shapely** - Geometric operations

### Optional Dependencies
- **matplotlib** - Basic plotting
- **plotly** - Interactive visualizations
- **jupyter** - Notebook environment for analysis

## Troubleshooting

### Common Installation Issues

??? failure "Empty data folder after cloning"
    
    If your `data/` folder is empty after cloning, the submodule wasn't initialized:
    
    ```bash
    # Initialize and update submodules
    git submodule update --init --recursive
    
    # Verify data is present
    ls data/
    # Should show: Cambodia, ECA, Ecuador, Global, Testkistan, etc.
    ```

??? failure "Submodule already exists error"
    
    If you get `fatal: 'data' already exists in the index` when trying to add submodules:
    
    ```bash
    # The submodule is already configured, just update it
    git submodule update --init --recursive
    
    # Don't run git submodule add again
    ```

??? failure "Conda environment creation hangs"
    
    The environment creation may get stuck in the "solving environment" step. Try:
    
    ```bash
    # Clear conda cache
    conda clean --all
    
    # Use libmamba solver (faster)
    conda install -n base conda-libmamba-solver
    conda config --set solver libmamba
    
    # Retry environment creation
    conda env create -f dsc-environment.yml
    ```

??? failure "Geospatial package installation fails"
    
    GeoPandas and related packages can be tricky to install. Solutions:
    
    ```bash
    # Install from conda-forge (recommended)
    conda install -c conda-forge geopandas
    
    # Or use mamba (faster conda alternative)
    mamba install geopandas
    ```

??? failure "Import errors after installation"
    
    If you get import errors when running DisruptSC:
    
    ```bash
    # Make sure environment is activated
    conda activate dsc
    
    # Reinstall in development mode
    pip install -e .
    
    # Check Python path
    python -c "import sys; print(sys.path)"
    ```

### Environment Issues

If you encounter environment conflicts:

```bash
# Remove existing environment
conda env remove -n dsc

# Recreate clean environment
conda env create -f dsc-environment.yml

# Alternative: create minimal environment and install manually
conda create -n dsc python=3.11
conda activate dsc
pip install -e .
```

## Development Installation

For developers contributing to DisruptSC:

```bash
# Clone with development dependencies and submodules
git clone --recurse-submodules https://github.com/worldbank/disrupt-sc.git
cd disrupt-sc

# Create development environment
conda env create -f dsc-environment.yml
conda activate dsc

# Install in development mode with test dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

## What's Next?

After successful installation:

1. **[Set up your data sources](data-setup.md)** - Configure input data
2. **[Run the quick start example](quick-start.md)** - Test your installation
3. **[Validate your inputs](validation.md)** - Ensure data quality

## Getting Help

If you're still having trouble:

- Review the common issues above
- Search [existing issues](https://github.com/worldbank/disrupt-sc/issues)
- Open a new issue with your error message and system details