# Reference

This section provides detailed reference information for DisruptSC users and developers.

## Quick Reference

### Command Line Interface
- **[CLI Reference](cli.md)** - Complete command-line options and usage patterns
- **Parameters Reference** - See User Guide for comprehensive parameter documentation

### Data Specifications  
- **File Formats** - See User Guide for detailed file format specifications
- **API Reference** - Code documentation (coming soon)

## Reference Categories

### User Reference
Essential information for model users:

- **Configuration** - Parameter files, command-line options
- **Data Formats** - Input file specifications and validation
- **Output Structure** - Result files and their contents
- **Error Messages** - Common errors and solutions

### Developer Reference  
Technical information for developers:

- **API Documentation** - Class and function reference
- **Architecture** - Internal model structure
- **Extension Points** - How to customize and extend
- **Testing** - Test suites and validation

## Common Lookups

### Quick Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `simulation_type` | `"initial_state"` | Type of simulation to run |
| `io_cutoff` | `0.01` | Input-output coefficient threshold |
| `utilization_rate` | `0.8` | Firm capacity utilization |
| `with_transport` | `true` | Enable transport modeling |
| `time_resolution` | `"day"` | Simulation time unit |

### File Format Quick Reference

| File | Format | Geometry | Required |
|------|--------|----------|----------|
| `mrio.csv` | CSV | - | ✅ |
| `sector_table.csv` | CSV | - | ✅ |
| `roads_edges.geojson` | GeoJSON | LineString | ✅ |
| `households.geojson` | GeoJSON | Point | ✅ |
| `maritime_edges.geojson` | GeoJSON | LineString | ⚠️ |

### Common Commands

```bash
# Basic simulation
python disruptsc/main.py Cambodia

# Validate inputs
python validate_inputs.py Cambodia

# Run with caching
python disruptsc/main.py Cambodia --cache same_transport_network_new_agents

# Custom parameters
python disruptsc/main.py Cambodia --io_cutoff 0.05 --duration 90
```

## Version Information

This documentation covers DisruptSC version 1.0.8 and later. For older versions, see the [changelog](https://github.com/ccolon/disrupt-sc/releases).

### Compatibility

- **Python**: 3.10 - 3.11
- **Data formats**: Stable since v1.0
- **Parameter structure**: Enhanced in v1.0.8
- **API**: Breaking changes marked in documentation

## Getting Help

### Documentation Issues
If you find errors or gaps in this reference:

1. Check the [GitHub Issues](https://github.com/ccolon/disrupt-sc/issues) for known documentation issues
2. Create a new issue with the "documentation" label
3. Suggest improvements via pull requests

### Model Issues
For model bugs or feature requests:

1. Search [existing issues](https://github.com/ccolon/disrupt-sc/issues)
2. Create detailed bug reports with reproducible examples
3. Include your parameter files and error messages

### Community Support
- **GitHub Discussions** - Questions and community help
- **Model documentation** - This reference and user guide
- **Code examples** - Tutorial and example repositories