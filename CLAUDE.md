# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Model

The main entry point is `src/disruptsc/main.py`. Run simulations with:

```bash
# Basic usage
python src/disruptsc/main.py <region>

# With caching options  
python src/disruptsc/main.py <region> --cache <cache_type>

# With custom parameters
python src/disruptsc/main.py <region> --duration 90 --io_cutoff 0.5
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

### Disruption System Architecture

The disruption system uses a factory pattern for flexible creation of different disruption types:

- **DisruptionFactory** - Central factory for creating disruptions from configuration
- **BaseDisruption** - Abstract base class for all disruption types
- **DisruptionContext** - Shared context data for disruption creation
- **DisruptionList** - Container for managing multiple disruptions

**Adding new disruption types:**
```python
def create_my_disruption(config: dict, context: DisruptionContext) -> MyDisruption:
    # Custom creation logic
    return MyDisruption(...)

DisruptionList.register_disruption_type("my_disruption", create_my_disruption)
```

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

Configuration uses YAML files in `config/parameters/`:
- `default.yaml` - Base parameters
- `user_defined_<region>.yaml` - Region-specific overrides

Key configuration areas:
- `simulation_type` - Controls execution mode (initial_state, disruption, criticality, etc.)
- `firm_data_type` - Data source (default: "mrio", alternative: "supplier-buyer network")
- `events` - Disruption scenarios for disruption simulations (note: "events" and "disruptions" refer to the same concept)
- File paths for region-specific input data

## Data Modes

DisruptSC supports two firm data modes:

### MRIO Mode (Default)
- **Usage**: Default mode, used as fallback for all cases
- **Data source**: Multi-Regional Input-Output tables
- **Households**: Generated from MRIO final demand
- **Countries**: Generated from MRIO trade flows
- **Supply chains**: Generated from IO technical coefficients
- **Configuration**: `firm_data_type: "mrio"` (or omitted)

### Supplier-Buyer Network Mode (Experimental)
- **Usage**: Alternative mode requiring specific network data
- **Data source**: Explicit supplier-buyer transaction tables
- **Households**: Still generated from MRIO final demand
- **Countries**: Still generated from MRIO trade flows  
- **Supply chains**: Generated from predefined transaction relationships
- **Configuration**: `firm_data_type: "supplier-buyer network"`
- **Required files**: `firm_table.csv`, `location_table.csv`, `transaction_table.csv`

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
Data organized by scope in `data/<scope>/` (or `input/<scope>/` for legacy):

```
data/{scope}/
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

Results saved to `output/<scope>/<timestamp>/`:
- `*_data.json` - Agent state timeseries
- `*_table.csv` - Tabular results  
- `*.geojson` - Geographic outputs
- `parameters.yaml` - Run configuration snapshot
- `exp.log` - Execution log

## Input File Validation

Validate input files before running simulations to catch errors early:

```bash
# Validate all input files for a scope
python validate_inputs.py <scope>

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

# Test specific scope inputs
python scripts/validate_inputs.py Cambodia
python scripts/validate_inputs.py ECA
```

## Version Management

Version is centrally managed in `src/disruptsc/_version.py`:

```python
# Check version programmatically
import disruptsc
print(disruptsc.__version__)

# Check version via CLI
python src/disruptsc/main.py --version
python -m disruptsc.model.input_validation --version
```

To update version: edit `src/disruptsc/_version.py` - setup.py automatically reads from this file.

**Recent changes (v1.0.8):**
- Simplified data modes: MRIO as default/fallback, supplier-buyer network as alternative
- Removed IO disaggregation mode complexity for cleaner codebase
- Households and countries now always use MRIO data for consistency
- Removed 12 unused parameters from Parameters class for cleaner codebase
- Updated dependencies with flexible version ranges
- Added comprehensive input validation system

## Development Notes

- Input validation system in `src/disruptsc/model/input_validation.py`
- Test suite covers common input file errors and edge cases
- Interactive development notebooks in `research/interactive/` folder
- Profiling enabled by default in main.py (cProfile)
- Logging configured via parameters, exports to timestamped folders
- Core dependencies: pandas, numpy, geopandas, networkx, scipy, shapely, PyYAML, tqdm (flexible version ranges)

## Performance Optimization - Logistics Routes

### Current Bottleneck Analysis
The logistics routes setup phase is the primary performance bottleneck during model initialization. Key issues:
- Sequential processing with `parallelized=False` in `model.py:495,505`
- Individual route calculations for each commercial link using NetworkX shortest path
- Limited effectiveness of route caching during bulk setup phase

### Spatial Clustering Optimization Framework

**Available Spatial Infrastructure:**
- Transport nodes with `lat`/`long` coordinates via `TransportNetwork._node[id]['lat/long']`
- Agents assigned to nodes using `od_point` via KDTree nearest neighbor (`find_nearest_node_id`)
- Distance calculations using `degrees_to_km` function
- Existing distance caching system in `TransportNetwork._distance_cache`

**Implementation Workflow:**

#### 1. Agent Spatial Clustering
```python
def create_agent_clusters(agents, transport_network, cluster_params):
    # Extract agent coordinates via their od_points
    agent_coords = [(agent.od_point, 
                    transport_network._node[agent.od_point]['lat'],
                    transport_network._node[agent.od_point]['long']) 
                   for agent in agents]
    
    # Apply clustering algorithm (DBSCAN or K-means)
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=cluster_params['max_distance_km'] / 111,  # degrees
                       min_samples=cluster_params['min_agents_per_cluster'])
    
    return group_agents_by_cluster(agents, clustering.labels_)
```

#### 2. Hub Identification Algorithm
```python
def identify_transport_hubs(transport_network, centrality_params):
    # Calculate multiple centrality measures
    centrality_scores = {
        'betweenness': nx.betweenness_centrality(transport_network),
        'closeness': nx.closeness_centrality(transport_network),
        'degree': nx.degree_centrality(transport_network),
        'eigenvector': nx.eigenvector_centrality(transport_network)
    }
    
    # Weighted composite score
    hub_scores = calculate_composite_hub_score(centrality_scores)
    return select_top_hubs(hub_scores, centrality_params['num_hubs'])
```

#### 3. Hierarchical Routing Workflow
- **Hub-to-Hub Backbone**: Pre-compute all hub-to-hub routes
- **Cluster-to-Hub Assignment**: Assign each agent cluster to nearest hub
- **Route Construction**: Agent → Hub → Hub → Agent routing pattern
- **Cache Integration**: Hierarchical cache structure with route segment reuse

#### 4. Core Data Structures
```python
class SpatialCluster:
    def __init__(self, cluster_id, agents, centroid_node, hub_node):
        self.cluster_id = cluster_id
        self.agents = agents
        self.centroid_node = centroid_node
        self.hub_node = hub_node
        self.internal_routes_cache = {}

class TransportHub:
    def __init__(self, node_id, centrality_score, economic_score):
        self.node_id = node_id
        self.centrality_score = centrality_score
        self.economic_score = economic_score
        self.connected_clusters = []
        self.hub_routes = {}  # Routes to other hubs
```

#### 5. Integration Points
- **Model Setup**: Add clustering/hub identification after agent location in `setup_logistic_routes`
- **Route Selection**: Replace direct routing with hierarchical routing in `choose_initial_routes`
- **Caching**: Extend existing cache with hierarchical structure in `caching_functions.py`
- **Parameters**: Add clustering/hub configuration to YAML parameter files

#### 6. Implementation Priority
1. **Enable parallelization** (immediate 2-4x speedup) - Change `parallelized=False` to `True`
2. **Bulk route pre-computation** - Collect all OD pairs before computing routes
3. **Spatial clustering** - Implement agent clustering and hub identification
4. **Hierarchical routing** - Implement hub-based routing logic
5. **Enhanced caching** - Hierarchical cache structure with segment reuse

**Files to modify:**
- `src/disruptsc/model/model.py` - Add clustering to `setup_logistic_routes`
- `src/disruptsc/agents/transport_mixin.py` - Modify `choose_initial_routes` for hierarchical routing
- `src/disruptsc/network/transport_network.py` - Add hub identification methods
- `src/disruptsc/model/caching_functions.py` - Extend caching for hierarchical routes
- Configuration files - Add clustering/hub parameters