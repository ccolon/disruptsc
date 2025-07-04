# Parameters

DisruptSC uses a hierarchical configuration system with YAML files. This guide explains all available parameters and their usage.

## Configuration System

### File Hierarchy

1. **`parameter/default.yaml`** - Base parameters (don't edit)
2. **`parameter/user_defined_<scope>.yaml`** - Scope-specific overrides

Only edit the user-defined files. Default parameters are loaded first, then overridden by user settings.

### Parameter Override

```yaml
# parameter/user_defined_Cambodia.yaml
simulation_type: "disruption"    # Override default
io_cutoff: 0.05                 # Override default
# Other parameters inherit from default.yaml
```

### Command Line Overrides

Key parameters can be overridden from command line:

```bash
python disruptsc/main.py Cambodia --io_cutoff 0.05 --duration 90
```

## Core Simulation Parameters

### Simulation Control

```yaml
# Simulation type and duration
simulation_type: "initial_state"  # See options below
t_final: 365                      # Simulation duration (time units)
time_resolution: "day"            # Time unit: "day", "week", "month"
epsilon_stop_condition: true      # Stop when equilibrium reached
```

**Simulation Types:**

| Type | Purpose | When to Use |
|------|---------|-------------|
| `initial_state` | Baseline analysis | Understanding normal operations |
| `disruption` | Single disruption | Testing specific scenarios |
| `disruption_mc` | Monte Carlo analysis | Statistical robustness |
| `criticality` | Infrastructure assessment | Finding critical links |
| `disruption-sensitivity` | Parameter sensitivity | Testing parameter robustness |
| `flow_calibration` | Transport calibration | Matching observed data |

### Scope and Regions

```yaml
scope: "Cambodia"                 # Main study region
```

## Data Configuration

### Data Sources

```yaml
# Data input mode
firm_data_type: "mrio"           # "mrio" or "supplier-buyer network"

# Monetary units
monetary_units_in_model: "mUSD"   # Model currency: "USD", "kUSD", "mUSD"
monetary_units_in_data: "USD"    # Data currency: "USD", "kUSD", "mUSD"

# File paths (relative to data folder)
filepaths:
  mrio: "Economic/mrio.csv"
  sector_table: "Economic/sector_table.csv"
  households_spatial: "Spatial/households.geojson"
  firms_spatial: "Spatial/firms.geojson"
  countries_spatial: "Spatial/countries.geojson"
  # Transport networks
  roads_edges: "Transport/roads_edges.geojson"
  maritime_edges: "Transport/maritime_edges.geojson"
  railways_edges: "Transport/railways_edges.geojson"
  # Additional files for supplier-buyer mode
  firm_table: "Economic/firm_table.csv"
  location_table: "Economic/location_table.csv"
  transaction_table: "Economic/transaction_table.csv"
```

### Data Filtering

```yaml
# Economic thresholds
io_cutoff: 0.01                  # Input-output coefficient threshold
cutoff_firm_output:
  value: 1000000                 # Minimum firm output
  unit: "USD"                    # Unit for threshold
cutoff_sector_output:
  value: 50000000                # Minimum sector output
  unit: "USD"
cutoff_household_demand:
  value: 100                     # Minimum household demand
  unit: "USD"

# Sector filtering
sectors_to_include: []           # Empty = include all
sectors_to_exclude: []           # Empty = exclude none
# Example: ["AGR", "MAN"] or ["SER_*"] (wildcards supported)

# Regional filtering  
countries_to_include: []         # Empty = include all trading partners
```

## Agent Parameters

### Firm Behavior

```yaml
# Production parameters
utilization_rate: 0.8            # Normal capacity utilization
capital_to_value_added_ratio: 4  # Capital intensity
inventory_restoration_time: 1    # Inventory rebuild speed (time units)

# Inventory management
inventory_duration_targets:
  default: 7                     # Days of inventory to maintain
  AGR: 3                        # Sector-specific overrides
  MAN: 14
  SER: 1

# Financial parameters
target_margin: 0.2               # Profit margin target
transport_share: 0.2             # Transport cost share of output
```

### Supply Chain Formation

```yaml
# Supplier selection
nb_suppliers_per_input: 1.5      # Average suppliers per input (1-2)
weight_localization_firm: 2.0    # Distance preference (higher = more local)
weight_localization_household: 1.5  # Household retailer distance preference

# Market behavior
adaptive_inventories: true       # Adjust inventory targets
adaptive_supplier_weight: true   # Change supplier preferences
rationing_mode: "equal"          # How to allocate scarce supplies
```

## Transport Parameters

### Transport Modeling

```yaml
# Transport system
with_transport: true             # Enable transport modeling
transport_modes: ["roads", "maritime", "railways"]  # Active modes
transport_to_households: false   # Model household transport explicitly
sectors_no_transport_network: ["SER", "UTI"]  # Service sectors

# Performance and routing
capacity_constraint: false       # Enable transport capacity limits
use_route_cache: true           # Cache routing calculations
route_optimization_weight: "time"  # Optimization criteria
congestion: false               # Enable congestion modeling
```

### Transport Economics

```yaml
# Cost parameters
price_increase_threshold: 0.5    # Maximum price increase tolerance

# Logistics parameters
logistics:
  nb_cost_profiles: 3            # Number of different cost profiles
  sector_types_to_shipment_method:
    agriculture: "bulk"
    manufacturing: "container" 
    service: "express"
```

## Disruption Parameters

### Event Configuration

```yaml
# Disruption events
events:
  - type: "transport_disruption"
    description_type: "edge_attributes"
    attribute: "highway"          # Edge attribute to match
    value: ["primary", "trunk"]   # Values indicating disruption
    start_time: 10               # When disruption starts
    duration: 20                 # How long it lasts
    
  - type: "capital_destruction"
    description_type: "region_sector_file"
    region_sector_filepath: "Disruption/earthquake_damage.csv"
    unit: "mUSD"
    reconstruction_market: true   # Enable reconstruction
    start_time: 5
```

### Recovery Parameters

```yaml
# Recovery modeling
recovery:
  transport_recovery_rate: 0.1   # Daily recovery rate (0-1)
  capital_recovery_rate: 0.05    # Capital rebuilding rate
  adaptive_recovery: true        # Priority-based recovery
```

### Criticality Analysis

```yaml
criticality:
  duration: 30                   # Days to simulate each disruption
  edges_to_test: "all"          # "all", "primary", or specific list
  metrics: ["production_loss", "welfare_loss"]  # Impact measures
```

## Performance Parameters

### Computational Settings

```yaml
# Execution control
logging_level: "INFO"            # "DEBUG", "INFO", "WARNING", "ERROR"
export_files: true              # Save detailed outputs
flow_data: true                 # Export transport flow data

# Parallel processing
parallelized: false             # Enable parallel route calculation
max_workers: 4                  # Number of parallel workers

# Memory management
cache_size: 1000                # Route cache size
batch_size: 100                 # Processing batch size
```

### Monte Carlo Settings

```yaml
# Monte Carlo analysis
mc_repetitions: 100             # Number of MC runs
mc_seed: 42                     # Random seed for reproducibility
mc_parallel: true              # Parallel MC execution
mc_output_aggregation: "summary"  # "full", "summary", "minimal"
```

### Sensitivity Analysis Settings

```yaml
# Parameter sensitivity analysis
simulation_type: "disruption-sensitivity"
sensitivity:
  io_cutoff: [0.01, 0.05, 0.1]                    # Economic threshold values
  utilization: [0.8, 0.9, 1.0]                    # Transport capacity utilization
  inventory_duration_targets.values.transport: [1, 3, 5]  # Inventory targets (nested)
  price_increase_threshold: [0.05, 0.1, 0.15]     # Price shock thresholds
```

**Sensitivity Configuration:**

- **Parameter specification:** List all values to test for each parameter
- **Nested parameters:** Use dot notation (e.g., `parent.child.property`)
- **Cartesian product:** All combinations are automatically generated
- **Output:** Single CSV file with results for each combination
- **No caching:** Each combination rebuilds the complete model

**Example with 3×3×3×3 = 81 combinations:**
```yaml
sensitivity:
  io_cutoff: [0.01, 0.05, 0.1]
  utilization: [0.8, 0.9, 1.0] 
  price_increase_threshold: [0.05, 0.1, 0.15]
  inventory_duration_targets.values.transport: [1, 3, 5]
```

## Advanced Parameters

### Model Calibration

```yaml
# Calibration targets
calibration:
  target_flows: "observed_flows.csv"  # Observed transport data
  target_prices: "price_data.csv"     # Market price data
  weight_flows: 0.7                   # Relative importance of flow matching
  weight_prices: 0.3                  # Relative importance of price matching
  max_iterations: 50                  # Calibration iterations
  tolerance: 0.01                     # Convergence tolerance
```

### Experimental Features

```yaml
# Advanced features (experimental)
explicit_service_firm: false    # Explicit service firm modeling
congestion_modeling: false      # Traffic congestion effects
price_dynamics: false          # Dynamic price adjustment
firm_entry_exit: false         # Firm birth/death processes
learning_effects: false        # Adaptive agent behavior
```

## Parameter Validation

### Automatic Validation

DisruptSC validates parameters on startup:

```python
# Example validation checks
assert 0 <= utilization_rate <= 1, "Utilization rate must be 0-1"
assert nb_suppliers_per_input >= 1, "Must have at least 1 supplier"
assert io_cutoff >= 0, "IO cutoff cannot be negative"
```

### Custom Validation

Add custom validation to your parameter files:

```yaml
# Parameter constraints (checked automatically)
_validation:
  io_cutoff:
    min: 0
    max: 1
    description: "Input-output coefficient threshold"
  utilization_rate:
    min: 0.1
    max: 1.0
    description: "Firm capacity utilization"
```

## Parameter Examples

### Baseline Configuration

```yaml
# parameter/user_defined_Cambodia.yaml
simulation_type: "initial_state"
t_final: 1
time_resolution: "day"
io_cutoff: 0.01
utilization_rate: 0.8
with_transport: true
capacity_constraint: false
```

### Disruption Scenario

```yaml
simulation_type: "disruption"
t_final: 90
events:
  - type: "transport_disruption"
    description_type: "edge_attributes"  
    attribute: "highway"
    value: ["primary"]
    start_time: 10
    duration: 30
with_transport: true
capacity_constraint: true
adaptive_inventories: true
```

### High-Performance Configuration

```yaml
# Large-scale model optimization
cutoff_firm_output:
  value: 5000000
  unit: "USD"
cutoff_sector_output:
  value: 100000000
  unit: "USD"
sectors_to_exclude: ["SER_*"]
transport_to_households: false
use_route_cache: true
parallelized: true
max_workers: 8
```

### Monte Carlo Analysis

```yaml
simulation_type: "disruption_mc"
mc_repetitions: 500
mc_parallel: true
mc_seed: 12345
events:
  - type: "transport_disruption"
    description_type: "random_edges"
    probability: 0.1
    start_time: 10
    duration: 20
```

## Parameter Tuning Guidelines

### Economic Realism

- **io_cutoff**: Start with 0.01, increase to reduce model size
- **utilization_rate**: 0.7-0.9 for most economies
- **target_margin**: 0.15-0.25 typical for most sectors
- **inventory_duration_targets**: 3-30 days depending on sector

### Computational Performance

- **Reduce model size**: Increase cutoff values
- **Speed up routing**: Enable route caching
- **Memory optimization**: Exclude service sectors if not needed
- **Parallel processing**: Enable for large models

### Disruption Realism

- **Start small**: Begin with short, localized disruptions
- **Gradual recovery**: Use realistic recovery rates
- **Multiple scenarios**: Test range of disruption severities
- **Validate impacts**: Compare with historical data when available

## Sensitivity Analysis

### Parameter Sensitivity Testing

```python
# Example sensitivity analysis
import itertools
import subprocess

# Parameters to test
io_cutoffs = [0.005, 0.01, 0.02, 0.05]
utilization_rates = [0.7, 0.8, 0.9]

# Run all combinations
for io_cutoff, util_rate in itertools.product(io_cutoffs, utilization_rates):
    cmd = [
        "python", "disruptsc/main.py", "Cambodia",
        "--io_cutoff", str(io_cutoff),
        "--utilization_rate", str(util_rate)
    ]
    subprocess.run(cmd)
```

### Key Sensitivity Parameters

1. **io_cutoff** - Affects model size and economic completeness
2. **utilization_rate** - Influences production capacity and resilience
3. **nb_suppliers_per_input** - Controls supply chain redundancy
4. **weight_localization** - Determines spatial trade patterns
5. **inventory_duration_targets** - Affects shock absorption capacity

## Troubleshooting

### Common Parameter Issues

!!! failure "Model too large"
    
    **Solution**: Increase cutoff parameters
    ```yaml
    cutoff_firm_output:
      value: 10000000  # Increase from 1000000
    io_cutoff: 0.05    # Increase from 0.01
    ```

!!! failure "Unrealistic results"
    
    **Solution**: Check economic parameters
    ```yaml
    utilization_rate: 0.8    # Not 0.95+
    target_margin: 0.2       # Not 0.5+
    transport_share: 0.2     # Not 0.8+
    ```

!!! failure "Slow performance"
    
    **Solution**: Enable performance optimizations
    ```yaml
    use_route_cache: true
    transport_to_households: false
    sectors_no_transport_network: ["SER", "UTI", "TRA"]
    ```

### Parameter Debugging

Enable detailed logging to debug parameter issues:

```yaml
logging_level: "DEBUG"
export_files: true
```

Check the execution log for parameter validation messages:

```bash
grep -i "parameter\|validation\|warning" output/Cambodia/latest/exp.log
```

## Best Practices

### Development Workflow

1. **Start simple** - Use default parameters initially
2. **Iterative refinement** - Change one parameter at a time
3. **Validate results** - Compare with known benchmarks
4. **Document changes** - Track parameter modifications
5. **Sensitivity testing** - Test parameter robustness

### Production Settings

1. **Lock parameters** - Use specific versions for production
2. **Archive configs** - Save parameter files with results
3. **Validate inputs** - Always validate before production runs
4. **Monitor performance** - Track runtime and memory usage
5. **Result validation** - Check output reasonableness

### Collaboration

1. **Standardize configs** - Use consistent parameter names
2. **Document rationale** - Explain parameter choices
3. **Version control** - Track parameter file changes
4. **Share configs** - Distribute validated parameter sets
5. **Peer review** - Have others review parameter choices