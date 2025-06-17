# CLI Reference

Complete command-line interface reference for DisruptSC.

## Basic Syntax

```bash
python disruptsc/main.py <scope> [options]
```

## Required Arguments

### Scope

```bash
python disruptsc/main.py <scope>
```

The scope argument specifies the region/case study to analyze.

**Requirements:**
- Must have data folder: `data/<scope>/` or `input/<scope>/`
- Must have parameter file: `parameter/user_defined_<scope>.yaml`

**Available scopes:**
- `Cambodia` - Southeast Asian economy
- `ECA` - Europe & Central Asia  
- `Ecuador` - South American economy
- `Global` - World economy
- `Testkistan` - Synthetic test case

## Options

### Caching Options

Control which components are rebuilt vs. reused:

```bash
--cache <cache_type>
```

| Cache Type | Description | Use Case |
|------------|-------------|----------|
| `same_transport_network_new_agents` | Reuse transport, rebuild agents | Testing agent parameters |
| `same_agents_new_sc_network` | Reuse agents, rebuild supply chains | Testing supplier selection |
| `same_sc_network_new_logistic_routes` | Reuse supply chains, rebuild routes | Testing transport parameters |
| `same_logistic_routes` | Reuse everything | Running disruption scenarios |

**Examples:**
```bash
# First run (no cache)
python disruptsc/main.py Cambodia

# Reuse transport network
python disruptsc/main.py Cambodia --cache same_transport_network_new_agents

# Reuse everything except routes
python disruptsc/main.py Cambodia --cache same_sc_network_new_logistic_routes
```

### Parameter Overrides

Override default parameters from command line:

#### Economic Parameters

```bash
--io_cutoff <float>              # Input-output coefficient threshold (default: 0.01)
--cutoff_firm_output_value <int> # Minimum firm output in model units
--cutoff_sector_output_value <int> # Minimum sector output
--utilization_rate <float>       # Firm capacity utilization (0-1)
--duration <int>                 # Simulation duration in time steps
```

#### Transport Parameters

```bash
--capacity_constraint <bool>     # Enable transport capacity limits
--transport_to_households <bool> # Model household transport
--with_transport <bool>          # Enable transport modeling
```

#### Data Parameters

```bash
--firm_data_type <string>        # "mrio" or "supplier-buyer network"
--monetary_units_in_model <string> # "USD", "kUSD", "mUSD"
--time_resolution <string>       # "day", "week", "month"
```

#### Filtering Parameters

```bash
--sectors_to_include <list>      # Only include specified sectors
--sectors_to_exclude <list>      # Exclude specified sectors
--countries_to_include <list>    # Only include specified countries
```

**Examples:**
```bash
# Economic parameters
python disruptsc/main.py Cambodia --io_cutoff 0.05 --utilization_rate 0.9

# Transport parameters  
python disruptsc/main.py Cambodia --capacity_constraint true --transport_to_households false

# Sector filtering
python disruptsc/main.py Cambodia --sectors_to_include "AGR,MAN" --sectors_to_exclude "SER"
```

### Simulation Control

```bash
--simulation_type <string>       # Simulation type (see below)
--config <path>                  # Use custom configuration file
--output_dir <path>              # Custom output directory
```

**Simulation types:**
- `initial_state` - Baseline equilibrium analysis
- `disruption` - Single disruption scenario
- `disruption_mc` - Monte Carlo disruption analysis  
- `criticality` - Infrastructure criticality assessment
- `flow_calibration` - Transport flow calibration

### Utility Options

```bash
--version                        # Show version information
--help                          # Show help message
--verbose                       # Enable verbose output
--quiet                         # Suppress non-error output
--debug                         # Enable debug mode
```

### Validation Options

```bash
--validate-only                 # Only run input validation, don't simulate
--skip-validation               # Skip input validation (not recommended)
--validation-level <string>     # "basic", "standard", "comprehensive"
```

## Complex Examples

### Development Workflow

```bash
# 1. Validate inputs
python validate_inputs.py Cambodia

# 2. Quick test run
python disruptsc/main.py Cambodia --simulation_type initial_state --duration 1

# 3. Full baseline with caching
python disruptsc/main.py Cambodia

# 4. Test disruption scenarios (fast with cache)
python disruptsc/main.py Cambodia --cache same_logistic_routes --simulation_type disruption
```

### Performance Optimization

```bash
# Reduce model size for faster runs
python disruptsc/main.py Cambodia \
  --cutoff_firm_output_value 5000000 \
  --cutoff_sector_output_value 100000000 \
  --sectors_to_exclude "SER" \
  --transport_to_households false
```

### Sensitivity Analysis

```bash
# Test different IO cutoffs
for cutoff in 0.005 0.01 0.02 0.05; do
  python disruptsc/main.py Cambodia \
    --io_cutoff $cutoff \
    --cache same_transport_network_new_agents
done
```

### Monte Carlo Analysis

```bash
# Run Monte Carlo simulation
python disruptsc/main.py Cambodia \
  --simulation_type disruption_mc \
  --duration 90 \
  --config parameter/cambodia_mc.yaml
```

## Configuration Files

### Parameter Files

Use custom configuration files:

```bash
# Use custom config
python disruptsc/main.py Cambodia --config my_custom_config.yaml

# Combine with overrides
python disruptsc/main.py Cambodia --config my_config.yaml --io_cutoff 0.05
```

### Environment Variables

Set environment variables for global configuration:

```bash
# Data path
export DISRUPT_SC_DATA_PATH=/path/to/data

# Output directory
export DISRUPT_SC_OUTPUT_PATH=/path/to/outputs

# Run simulation
python disruptsc/main.py Cambodia
```

## Exit Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | Success | Simulation completed successfully |
| 1 | General error | Unspecified error occurred |
| 2 | Invalid arguments | Command-line arguments invalid |
| 3 | Missing data | Required input files not found |
| 4 | Validation error | Input validation failed |
| 5 | Memory error | Insufficient memory |
| 6 | Timeout | Simulation exceeded time limit |

## Output Control

### Verbosity Levels

```bash
# Minimal output
python disruptsc/main.py Cambodia --quiet

# Standard output (default)
python disruptsc/main.py Cambodia

# Verbose output
python disruptsc/main.py Cambodia --verbose

# Debug output (very detailed)
python disruptsc/main.py Cambodia --debug
```

### Output Redirection

```bash
# Save all output to file
python disruptsc/main.py Cambodia > simulation.log 2>&1

# Save only errors
python disruptsc/main.py Cambodia 2> errors.log

# Real-time monitoring
python disruptsc/main.py Cambodia --verbose | tee simulation.log
```

## Parallel Execution

### Multiple Scopes

```bash
# Run multiple scopes in parallel
for scope in Cambodia Ecuador Global; do
  python disruptsc/main.py $scope &
done
wait  # Wait for all to complete
```

### Parameter Sweeps

```bash
# Parallel parameter sweep
#!/bin/bash
for cutoff in 0.01 0.02 0.05; do
  for util in 0.7 0.8 0.9; do
    python disruptsc/main.py Cambodia \
      --io_cutoff $cutoff \
      --utilization_rate $util &
  done
done
wait
```

## Debugging and Troubleshooting

### Debug Mode

```bash
# Enable full debugging
python disruptsc/main.py Cambodia --debug --verbose

# Check specific components
python disruptsc/main.py Cambodia --debug --validate-only
```

### Memory Monitoring

```bash
# Monitor memory usage (Linux/Mac)
time -v python disruptsc/main.py Cambodia

# Monitor with htop/top
htop &
python disruptsc/main.py Cambodia
```

### Profiling

```bash
# Profile execution
python -m cProfile -o profile.prof disruptsc/main.py Cambodia

# Analyze profile
python -c "import pstats; pstats.Stats('profile.prof').sort_stats('cumulative').print_stats(20)"
```

## Integration with Other Tools

### Shell Scripts

```bash
#!/bin/bash
# run_simulation.sh

SCOPE=$1
if [ -z "$SCOPE" ]; then
    echo "Usage: $0 <scope>"
    exit 1
fi

# Validate first
echo "Validating inputs for $SCOPE..."
python validate_inputs.py $SCOPE || exit 1

# Run simulation
echo "Running simulation for $SCOPE..."
python disruptsc/main.py $SCOPE --verbose

echo "Simulation complete for $SCOPE"
```

### Python Scripts

```python
import subprocess
import sys

def run_simulation(scope, **kwargs):
    """Run DisruptSC simulation with parameters."""
    cmd = ["python", "disruptsc/main.py", scope]
    
    # Add parameter overrides
    for param, value in kwargs.items():
        cmd.extend([f"--{param}", str(value)])
    
    # Run simulation
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    return result.stdout

# Example usage
output = run_simulation("Cambodia", io_cutoff=0.05, duration=90)
print(output)
```

### Makefiles

```makefile
# Makefile for DisruptSC simulations

SCOPES = Cambodia Ecuador Global
OUTPUT_DIR = results

.PHONY: all validate simulate clean

all: validate simulate

validate:
	@for scope in $(SCOPES); do \
		echo "Validating $$scope..."; \
		python validate_inputs.py $$scope; \
	done

simulate:
	@for scope in $(SCOPES); do \
		echo "Simulating $$scope..."; \
		python disruptsc/main.py $$scope --verbose; \
	done

clean:
	rm -rf $(OUTPUT_DIR)/*
	rm -rf tmp/*

# Individual scope targets
cambodia:
	python disruptsc/main.py Cambodia --verbose

ecuador:
	python disruptsc/main.py Ecuador --verbose
```

## Best Practices

### Command Line Usage

1. **Always validate first** - Use `validate_inputs.py` before long runs
2. **Use caching** - Speed up development with `--cache` options
3. **Start small** - Test with short durations before full runs
4. **Monitor resources** - Watch memory and CPU usage
5. **Save outputs** - Redirect logs and capture results

### Parameter Management

1. **Use config files** - Avoid long command lines
2. **Document overrides** - Record why you changed defaults
3. **Version parameters** - Track parameter file changes
4. **Test incrementally** - Change one parameter at a time
5. **Validate results** - Check output reasonableness

### Production Runs

1. **Lock versions** - Use specific model and data versions
2. **Archive configs** - Save parameter files with results
3. **Monitor execution** - Use logging and monitoring tools
4. **Backup results** - Copy important outputs to permanent storage
5. **Document runs** - Record what was done and why