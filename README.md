# DisruptSC

[![Version](https://img.shields.io/badge/version-1.1.4-blue)](https://github.com/ccolon/disrupt-sc/releases/tag/v1.1.4)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://ccolon.github.io/disrupt-sc)
[![License](https://img.shields.io/github/license/ccolon/disrupt-sc)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![GitHub Issues](https://img.shields.io/github/issues/ccolon/disrupt-sc)](https://github.com/ccolon/disrupt-sc/issues)

**Spatial agent-based model for supply chain disruption analysis**

_DisruptSC_ simulates economic impacts of supply chain disruptions using spatial networks, multi-regional input-output data, and agent-based modeling. Analyze transport infrastructure failures, natural disasters, and other disruptions across countries and regions.

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/ccolon/disrupt-sc.git
cd disrupt-sc

# Create environment
conda env create -f dsc-environment.yml
conda activate dsc

# Setup data (choose one option)
# OR mkdir input                                # Option 1: Local folder
# OR git submodule add <data-repo-url> data     # Option 2: Git submodule (private repo, invitation-only)
```

### Run Your First Simulation
```bash
# Validate your input data first
python validate_inputs.py Cambodia

# Run baseline simulation
python disruptsc/main.py Cambodia

# Run disruption scenario
python disruptsc/main.py Cambodia --simulation_type disruption
```

## Documentation

**Full documentation available at: [https://ccolon.github.io/disrupt-sc](https://ccolon.github.io/disrupt-sc)**

- **[Getting Started](https://ccolon.github.io/disrupt-sc/getting-started/)** - Installation, data setup, first simulation
- **[User Guide](https://ccolon.github.io/disrupt-sc/user-guide/)** - Parameters, data modes, input/output files
- **[Tutorials](https://ccolon.github.io/disrupt-sc/tutorials/)** - Step-by-step examples and workflows
- **[Architecture](https://ccolon.github.io/disrupt-sc/architecture/)** - Model design, agents, networks, disruptions

## Key Features

- **Spatial Networks**: Multimodal transport infrastructure
- **Economic Agents**: Firms, households, and countries with spatial disaggregation
- **Disruption Scenarios**: Transport failures, capital destruction, productivity shock
- **Input Validation**: Comprehensive data quality checks before simulation

## Contributing & Support

- **Issues**: [Report bugs or request features](https://github.com/ccolon/disrupt-sc/issues)
- **Discussions**: [Ask questions or share ideas](https://github.com/ccolon/disrupt-sc/discussions)
- **Get involved**: See [contributor guidelines](https://ccolon.github.io/disrupt-sc/contacts/)


## Citation

If you use _DisruptSC_, please cite:

Colon, C., Hallegatte, S., & Rozenberg, J. (2021). Criticality analysis of a country’s transport network via an agent-based supply chain model. Nature Sustainability, 4(3), 209-215.

```bibtex
@article{colon2020disruptsc,
  author  = {Celian Colon and Stephane Hallegatte and Julie Rozenberg},
  title   = {Criticality analysis of a country’s transport network via an agent-based supply chain model},
  journal = {Nature Sustainability},
  volume  = {4},
  pages   = {209--215},
  year    = {2021},
  doi     = {10.1038/s41893-020-00649-4},
  url     = {https://www.nature.com/articles/s41893-020-00649-4}
}
```
```bibtex
@software{disruptsc2024,
  title={DisruptSC: Spatial Agent-Based Model for Supply Chain Disruption Analysis},
  author={Celian Colon},
  year={2024},
  url={https://github.com/ccolon/disrupt-sc}
}
```