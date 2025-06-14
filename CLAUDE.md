# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Model

The main entry point is `disruptsc/main.py`. Run simulations with:

```bash
# Basic usage
python disruptsc/main.py <region>

# With caching options  
python disruptsc/main.py <region> --cache <cache_type>

# With custom parameters
python disruptsc/main.py <region> --duration 90 --io_cutoff 0.5
```

**Regions available:** Cambodia, ECA, Ecuador, Global, Testkistan

**Cache options:** 
- `same_transport_network_new_agents` - Reuse transport network, rebuild agents
- `same_agents_new_sc_network` - Reuse agents, rebuild supply chain network  
- `same_sc_network_new_logistic_routes` - Reuse supply chain, rebuild logistics
- `same_logistic_routes` - Reuse everything

## Environment Setup

```bash
# Create conda environment
conda env create -f dsc-environment.yml
conda activate dsc

# Alternative: install via pip
pip install -e .

# Setup data repository (choose one option):

# Option 1: Git submodule (recommended)
git submodule add <your-private-data-repo-url> data
git submodule update --init

# Option 2: Environment variable
export DISRUPT_SC_DATA_PATH=/path/to/your/data/folder

# Option 3: Legacy - keep data in input/ folder (fallback)
```

**Python requirement:** 3.10-3.11 (supports Python 3.10 and 3.11)

## Architecture Overview

DisruptSC is a spatial agent-based model simulating supply chain disruptions. Key architectural components:

### Core Model Flow
1. **Transport Network Setup** - Load/build infrastructure (roads, maritime, railways)
2. **Agent Creation** - Firms, households, countries based on economic data
3. **Supply Chain Network** - Commercial relationships via MRIO or IO disaggregation  
4. **Logistics Routes** - Optimize transport paths between agents
5. **Simulation Execution** - Run scenarios based on `simulation_type` parameter

### Key Classes & Relationships
- `Model` - Main orchestrator in `model/model.py`
- `Mrio` - Multi-regional input-output data (extends pandas DataFrame)
- `ScNetwork` - Supply chain relationships (extends NetworkX DiGraph)
- `Agents` - Collections of firms, households, countries
- `TransportNetwork` - Infrastructure graph with spatial data
- `Simulation` - Execution engine with data collection

### Module Organization
- `agents/` - Economic actors (firms, households, countries)
- `model/` - Core logic, builders, caching, validation
- `network/` - Supply chain, transport, MRIO data structures  
- `simulation/` - Execution engine and event handling
- `disruption/` - Disruption types and recovery mechanisms

## Configuration

Configuration uses YAML files in `parameter/`:
- `default.yaml` - Base parameters
- `user_defined_<region>.yaml` - Region-specific overrides

Key configuration areas:
- `simulation_type` - Controls execution mode (initial_state, disruption, criticality, etc.)
- `firm_data_type` - Data source ("mrio" vs "supplier-buyer network")  
- `events` - Disruption scenarios for disruption simulations
- File paths for region-specific input data

## Input Data Structure

**Repository Separation:**
The model code and data are now separated into different repositories:
- **disrupt-sc** (public): Model code, configuration, documentation
- **disrupt-sc-data** (private): Input data files

**Data Location Options:**
1. **Git Submodule** (recommended): `git submodule add <data-repo-url> data`
2. **Environment Variable**: Set `DISRUPT_SC_DATA_PATH=/path/to/data`
3. **Legacy**: Keep data in `input/` folder (fallback)

**Note:** Large data files (>50MB) may trigger GitHub warnings. Consider migrating to Git Large File Storage (Git LFS) if file sizes become problematic or if frequent updates to large files are needed.

**Data Structure:**
Data organized by region in `data/<region>/` (or `input/<region>/` for legacy):

```
data/{region}/
├── Economic/        # MRIO tables, sector definitions
├── Transport/      # Infrastructure GeoJSON files
└── Spatial/      # Geographic disaggregation data
```

## Simulation Types

- `initial_state` - Baseline equilibrium analysis
- `disruption` - Single disruption scenario  
- `disruption_mc` - Monte Carlo disruption analysis
- `criticality` - Infrastructure edge criticality assessment
- `flow_calibration` - Calibrate transport flows to observed data

## Performance & Caching

Model uses pickle-based caching in `tmp/` folder for expensive operations:
- Transport network construction
- Agent creation and placement
- Supply chain network generation  
- Logistics route optimization

Use `--cache` parameter to selectively reuse cached components across runs.

## Output Structure

Results saved to `output/<region>/<timestamp>/`:
- `*_data.json` - Agent state timeseries
- `*_table.csv` - Tabular results  
- `*.geojson` - Geographic outputs
- `parameters.yaml` - Run configuration snapshot
- `exp.log` - Execution log

## Input File Validation

Validate input files before running simulations to catch errors early:

```bash
# Validate all input files for a region
python validate_inputs.py <region>

# Example
python validate_inputs.py Cambodia
```

**What gets validated:**
- File existence and readability
- Required columns in sector tables, MRIO data
- Data types and value ranges (no negative outputs, etc.)
- Transport network geometry (LineString required)
- MRIO table balance and structure
- Parameter consistency (monetary units, cutoffs)

**Integration with model:**
```python
from disruptsc.model.input_validation import validate_inputs
from disruptsc.parameters import Parameters

parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, scope)
is_valid, errors, warnings = validate_inputs(parameters)

if not is_valid:
    print("Errors found:", errors)
    exit(1)
```

## Testing

```bash
# Run input validation tests
python test_input_validation.py

# Test specific region inputs
python validate_inputs.py Cambodia
python validate_inputs.py ECA
```

## Version Management

Version is centrally managed in `disruptsc/_version.py`:

```python
# Check version programmatically
import disruptsc
print(disruptsc.__version__)

# Check version via CLI
python disruptsc/main.py --version
python -m disruptsc.model.input_validation --version
```

To update version: edit `disruptsc/_version.py` - setup.py automatically reads from this file.

**Recent changes (v1.0.8):**
- Removed 12 unused parameters from Parameters class for cleaner codebase
- Updated dependencies with flexible version ranges
- Added comprehensive input validation system

## Development Notes

- Input validation system in `disruptsc/model/input_validation.py`
- Test suite covers common input file errors and edge cases
- Interactive development notebooks in `interactive/` folder
- Profiling enabled by default in main.py (cProfile)
- Logging configured via parameters, exports to timestamped folders
- Core dependencies: pandas, numpy, geopandas, networkx, scipy, shapely, PyYAML, tqdm (flexible version ranges)