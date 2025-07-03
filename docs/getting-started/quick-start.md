# Quick Start

This guide will get you running your first DisruptSC simulation in just a few minutes.

## Prerequisites

Before starting, ensure you have:

- ✅ [Installed DisruptSC](installation.md)
- ✅ [Set up your data sources](data-setup.md)
- ✅ Activated the conda environment: `conda activate dsc`

## Available Regions

DisruptSC comes with several pre-configured regions:

| Region | Description | Scale | Use Case |
|--------|-------------|--------|----------|
| **Cambodia** | Southeast Asian economy | National | Regional trade analysis |
| **ECA** | Europe & Central Asia | Multi-country | Cross-border impacts |
| **Ecuador** | South American economy | National | Natural disaster studies |
| **Global** | World economy | International | Global supply chains |
| **Testkistan** | Synthetic test case | Small | Learning and testing |

## Your First Simulation

### Step 1: Validate Your Data

Before running simulations, validate your input data:

```bash
# Validate inputs for Cambodia
python validate_inputs.py Cambodia
```

You should see:
```
✅ Validation successful for Cambodia
All required files found and validated
Ready to run simulations
```

### Step 2: Run Basic Simulation

Start with an initial state analysis:

```bash
# Run baseline simulation for Cambodia
python disruptsc/main.py Cambodia
```

This will:

1. **Setup transport network** - Load roads, maritime, and other infrastructure
2. **Create agents** - Generate firms, households, and countries
3. **Build supply chains** - Connect buyers and suppliers
4. **Optimize routes** - Find efficient transport paths
5. **Initialize equilibrium** - Set baseline economic conditions
6. **Run simulation** - Execute the model and collect results

### Step 3: Check Results

Results are saved in timestamped folders:

```bash
# Navigate to results
ls output/Cambodia/

# View the latest results
cd output/Cambodia/$(ls -t output/Cambodia/ | head -1)
ls -la
```

**Key output files:**
- `firm_data.json` - Firm state over time
- `household_data.json` - Household consumption patterns
- `transport_edges_with_flows_0.geojson` - Transport flows visualization
- `parameters.yaml` - Configuration used for this run
- `exp.log` - Detailed execution log

## Advanced Usage

### Simulation Types

Configure different simulation types in your parameter file:

```yaml
# parameter/user_defined_Cambodia.yaml
simulation_type: "initial_state"  # Options below
```

**Available simulation types:**

| Type | Purpose | When to Use |
|------|---------|-------------|
| `initial_state` | Baseline equilibrium | Understanding normal operations |
| `disruption` | Single disruption scenario | Testing specific events |
| `criticality` | Infrastructure assessment | Finding critical links |
| `flow_calibration` | Transport calibration | Matching observed data |

### Monte Carlo Simulations

Control Monte Carlo runs with the `mc_repetitions` parameter:

```yaml
# In parameter file
mc_repetitions: 10               # Run 10 iterations
simulation_type: "disruption"    # Base simulation type
```

**Behavior:**
- **`mc_repetitions = 0`**: Single run with full output files
- **`mc_repetitions ≥ 1`**: Multiple runs with CSV summary only

### Caching for Performance

Speed up repeated runs with caching:

```bash
# First run (builds everything)
python disruptsc/main.py Cambodia

# Reuse transport network, rebuild agents
python disruptsc/main.py Cambodia --cache same_transport_network_new_agents

# Reuse agents, rebuild supply chains
python disruptsc/main.py Cambodia --cache same_agents_new_sc_network

# Reuse supply chains, rebuild routes
python disruptsc/main.py Cambodia --cache same_sc_network_new_logistic_routes

# Reuse everything
python disruptsc/main.py Cambodia --cache same_logistic_routes
```

### Custom Parameters

Override default parameters:

```bash
# Run with custom duration and cutoffs
python disruptsc/main.py Cambodia --duration 90 --io_cutoff 0.5

# See all available options
python disruptsc/main.py Cambodia --help
```

## Disruption Scenarios

### Simple Transport Disruption

Create a basic disruption scenario by editing your parameter file:

```yaml
# parameter/user_defined_Cambodia.yaml
simulation_type: "disruption"

events:
  - type: "transport_disruption"
    description_type: "edge_attributes"
    attribute: "highway_type"
    value: ["primary"]
    start_time: 10
    duration: 20
```

This disrupts primary highways from time step 10 for 20 time steps.

### Capital Destruction Event

Model economic impacts of disasters:

```yaml
events:
  - type: "capital_destruction"
    description_type: "region_sector_file"
    region_sector_filepath: "Disruption/earthquake_damage.csv"
    unit: "mUSD"
    reconstruction_market: true
    start_time: 5
```

## Understanding Results

### Economic Impacts

Key metrics to analyze:

- **Production losses** - Reduced output by firms
- **Consumption losses** - Unmet household demand  
- **Price changes** - Market adjustments
- **Regional effects** - Spatial distribution of impacts

### Visualization

Use the generated GeoJSON files for mapping:

```python
import geopandas as gpd
import matplotlib.pyplot as plt

# Load transport flows
flows = gpd.read_file('output/Cambodia/.../transport_edges_with_flows_0.geojson')

# Plot flow intensity
flows.plot(column='flow_volume', linewidth=2, cmap='Reds')
plt.title('Transport Flow Intensity')
plt.show()
```

## Common Issues

### Performance Tips

!!! tip "Speed up large models"
    
    ```bash
    # Reduce firm count with cutoffs
    python disruptsc/main.py Cambodia --cutoff_firm_output_value 1000000
    
    # Filter small sectors
    python disruptsc/main.py Cambodia --cutoff_sector_output_value 50000000
    
    # Use caching for development
    python disruptsc/main.py Cambodia --cache same_transport_network_new_agents
    ```

### Memory Issues

!!! warning "Large memory usage"
    
    If you encounter memory issues:
    
    - Increase system memory
    - Reduce model scope/resolution
    - Use firm/sector filtering
    - Consider running on cloud instances

### Debugging

!!! info "Troubleshooting runs"
    
    ```bash
    # Check execution log
    tail -f output/Cambodia/.../exp.log
    
    # Validate inputs first
    python validate_inputs.py Cambodia
    
    # Run with verbose output
    python disruptsc/main.py Cambodia --verbose
    ```

## Next Steps

After your first successful simulation:

1. **[Explore parameters](../user-guide/parameters.md)** - Customize simulation settings
2. **[Learn data modes](../user-guide/data-modes.md)** - Choose MRIO vs network data
3. **[Study output files](../user-guide/output-files.md)** - Detailed output analysis
4. **[Understand architecture](../architecture/overview.md)** - Learn how the model works

## Example Workflow

A typical research workflow:

```bash
# 1. Validate data
python validate_inputs.py Cambodia

# 2. Run baseline
python disruptsc/main.py Cambodia

# 3. Test disruption scenarios
python disruptsc/main.py Cambodia --simulation_type disruption

# 4. Analyze results
python analysis_script.py output/Cambodia/latest/

# 5. Run sensitivity analysis
for cutoff in 0.1 0.5 1.0; do
    python disruptsc/main.py Cambodia --io_cutoff $cutoff
done
```

Congratulations! You've successfully run your first DisruptSC simulation. 🎉