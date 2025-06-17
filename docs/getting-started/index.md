# Getting Started

Welcome to DisruptSC! This section will help you get up and running with the model quickly.

## Prerequisites

Before you begin, make sure you have:

- **Python 3.10 or 3.11** installed
- **Git** for repository management
- **Access to input data** (see [Data Setup](data-setup.md))

## Setup Process

Follow these steps to get DisruptSC running on your system:

1. **[Installation](installation.md)** - Set up the environment and dependencies
2. **[Data Setup](data-setup.md)** - Configure input data sources  
3. **[Quick Start](quick-start.md)** - Run your first simulation
4. **[Input Validation](validation.md)** - Verify your data is correctly formatted

## What's Next?

After completing the setup:

- Explore the **[User Guide](../user-guide/index.md)** for detailed usage instructions
- Try the **[Tutorials](../tutorials/index.md)** for hands-on examples
- Read about the **[Architecture](../architecture/index.md)** to understand how the model works

## Common Issues

??? question "Environment setup problems"
    
    If you encounter issues with conda environment creation, try:
    
    ```bash
    # Clear conda cache
    conda clean --all
    
    # Create environment with explicit solver
    conda env create -f dsc-environment.yml --solver=libmamba
    ```

??? question "Data path not found"
    
    Make sure your data path is correctly set:
    
    ```bash
    # Check if environment variable is set
    echo $DISRUPT_SC_DATA_PATH
    
    # Or verify data folder exists
    ls -la data/
    ```

??? question "Import errors when running"
    
    Ensure you've installed the package in development mode:
    
    ```bash
    conda activate dsc
    pip install -e .
    ```