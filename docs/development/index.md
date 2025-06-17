# Development

This section provides guidance for developers working with DisruptSC, including setup instructions, coding standards, and integration with development tools.

## Development Environment

### Prerequisites

- **Python 3.10-3.11** - DisruptSC supports Python 3.10 and 3.11
- **Git** - Version control system
- **conda** or **pip** - Package management
- **IDE/Editor** - VS Code, PyCharm, or similar

### Environment Setup

#### Using conda (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd disrupt-sc

# Create conda environment
conda env create -f dsc-environment.yml
conda activate dsc

# Verify installation
python -c "import disruptsc; print(disruptsc.__version__)"
```

#### Using pip

```bash
# Clone repository
git clone <repository-url>
cd disrupt-sc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy
```

### Data Setup

Set up the data repository:

```bash
# Option 1: Git submodule (recommended)
git submodule add <your-private-data-repo-url> data
git submodule update --init

# Option 2: Environment variable
export DISRUPT_SC_DATA_PATH=/path/to/your/data/folder

# Option 3: Legacy - keep data in input/ folder
mkdir input
# Copy data files to input/
```

## Development Workflow

### Testing

DisruptSC uses pytest for testing:

```bash
# Run all tests
pytest

# Run specific test file
pytest test_input_validation.py

# Run with coverage
pytest --cov=disruptsc --cov-report=html

# Run integration tests
pytest tests/integration/

# Run performance tests
pytest tests/performance/ --benchmark-only
```

### Code Quality

#### Formatting with Black

```bash
# Format all Python files
black disruptsc/

# Check formatting
black --check disruptsc/

# Format specific file
black disruptsc/main.py
```

#### Linting with flake8

```bash
# Check code style
flake8 disruptsc/

# Ignore specific errors
flake8 --ignore=E501,W503 disruptsc/
```

#### Type Checking with mypy

```bash
# Check types
mypy disruptsc/

# Check specific module
mypy disruptsc/model/
```

### Input Validation

Always validate inputs before running simulations:

```bash
# Validate all input files for a scope
python validate_inputs.py Cambodia

# Comprehensive validation
python validate_inputs.py Cambodia --comprehensive

# Validate specific file types
python validate_inputs.py Cambodia --check-economic --check-transport
```

## Working with Claude Code

DisruptSC is designed to work seamlessly with Claude Code (claude.ai/code) for AI-assisted development.

### Claude Code Integration

The `CLAUDE.md` file contains specific guidance for Claude Code:

- **Running commands** - How to execute simulations
- **Environment setup** - Installation and configuration
- **Architecture overview** - Core model components
- **Configuration system** - Parameter management
- **Data modes** - MRIO vs supplier-buyer networks
- **Performance optimization** - Caching and scaling

### Key Claude Code Commands

```bash
# Basic simulation
python disruptsc/main.py Cambodia

# With caching for development
python disruptsc/main.py Cambodia --cache same_transport_network_new_agents

# Parameter customization
python disruptsc/main.py Cambodia --io_cutoff 0.05 --duration 90

# Input validation
python validate_inputs.py Cambodia
```

### Claude Code Best Practices

1. **Always validate inputs** before making changes
2. **Use caching** for iterative development
3. **Test incrementally** - change one parameter at a time
4. **Monitor performance** - watch memory and execution time
5. **Document changes** - update parameters and reasoning

## Contributing Guidelines

### Code Standards

#### Python Style Guide

Follow PEP 8 with these specific conventions:

```python
# Function names: snake_case
def calculate_transport_cost(distance, mode):
    pass

# Class names: PascalCase
class TransportNetwork:
    pass

# Constants: UPPER_CASE
DEFAULT_IO_CUTOFF = 0.01

# Private methods: leading underscore
def _internal_helper(self):
    pass
```

#### Documentation Standards

```python
def complex_function(param1: str, param2: Optional[float] = None) -> Dict[str, Any]:
    """
    Brief description of function purpose.
    
    Longer description explaining the function's behavior, assumptions,
    and any important implementation details.
    
    Args:
        param1: Description of first parameter
        param2: Description of optional parameter with default behavior
        
    Returns:
        Dictionary containing results with keys:
        - 'result': Main computation result
        - 'metadata': Additional information about computation
        
    Raises:
        ValueError: When param1 is invalid
        NetworkError: When network operations fail
        
    Example:
        >>> result = complex_function("test", 1.5)
        >>> print(result['result'])
        42
    """
    pass
```

### Git Workflow

#### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

#### Commit Messages

```bash
# Format: type(scope): description
feat(agents): add household consumption adaptation
fix(network): resolve route caching memory leak
docs(api): update parameter documentation
refactor(simulation): simplify time step execution
```

#### Pull Request Process

1. **Create feature branch** from main
2. **Implement changes** following code standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run full test suite** to ensure compatibility
6. **Submit pull request** with clear description

### Testing Guidelines

#### Unit Tests

```python
import pytest
from disruptsc.agents import Firm

def test_firm_creation():
    """Test basic firm creation and initialization."""
    firm = Firm(
        pid="test_firm",
        sector="AGR_TEST",
        production_capacity=1000
    )
    
    assert firm.pid == "test_firm"
    assert firm.sector == "AGR_TEST"
    assert firm.production_capacity == 1000
    assert firm.production_current == 0

def test_firm_production():
    """Test firm production logic."""
    firm = Firm(pid="test", sector="MAN_TEST", production_capacity=1000)
    
    # Set production target
    firm.set_production_target(800)
    assert firm.production_target == 800
    
    # Execute production
    firm.produce()
    assert firm.production_current <= 800
    assert firm.production_current >= 0
```

#### Integration Tests

```python
def test_model_initialization():
    """Test complete model initialization workflow."""
    from disruptsc.model import Model
    from disruptsc.parameters import Parameters
    
    # Load test parameters
    params = Parameters.load_test_parameters("Testkistan")
    
    # Initialize model
    model = Model(params)
    model.initialize()
    
    # Verify model state
    assert len(model.agents.firms) > 0
    assert len(model.agents.households) > 0
    assert model.networks.transport_network.number_of_nodes() > 0
    assert model.networks.sc_network.number_of_edges() > 0
```

### Performance Guidelines

#### Profiling

```python
import cProfile
import pstats

# Profile simulation execution
profiler = cProfile.Profile()
profiler.enable()

# Run simulation
simulation.run()

profiler.disable()

# Analyze results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

#### Memory Monitoring

```python
import psutil
import gc

def monitor_memory():
    """Monitor memory usage during execution."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
    print(f"VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
    
    # Force garbage collection
    collected = gc.collect()
    print(f"Garbage collected: {collected} objects")
```

#### Optimization Best Practices

1. **Use vectorized operations** with NumPy/Pandas
2. **Cache expensive calculations** (routes, distances)
3. **Minimize object creation** in tight loops
4. **Use appropriate data structures** (sets for membership, dicts for lookups)
5. **Profile before optimizing** - measure actual bottlenecks

## Advanced Development

### Custom Agent Development

```python
from disruptsc.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    """Custom agent type with specialized behavior."""
    
    def __init__(self, pid: str, **kwargs):
        super().__init__(pid, **kwargs)
        self.custom_attribute = kwargs.get('custom_attribute', 0)
        
    def update(self, t: int, **kwargs):
        """Custom update logic."""
        # Implement custom behavior
        self.custom_logic(t)
        
        # Call parent update
        super().update(t, **kwargs)
        
    def custom_logic(self, t: int):
        """Implement custom agent logic."""
        pass
```

### Custom Disruption Development

```python
from disruptsc.disruption.disruption import BaseDisruption

def create_custom_disruption(config: dict, context) -> 'CustomDisruption':
    """Factory function for custom disruption."""
    return CustomDisruption(
        start_time=config['start_time'],
        duration=config.get('duration'),
        custom_param=config.get('custom_param', 1.0)
    )

class CustomDisruption(BaseDisruption):
    """Custom disruption implementation."""
    
    def __init__(self, start_time: int, custom_param: float, **kwargs):
        super().__init__(start_time, **kwargs)
        self.custom_param = custom_param
        
    def apply(self, context):
        """Apply custom disruption effects."""
        # Implement disruption logic
        pass
        
    def remove(self, context):
        """Remove disruption effects."""
        # Implement recovery logic
        pass

# Register the custom disruption
from disruptsc.disruption.disruption import DisruptionFactory
DisruptionFactory.register_disruption_type(
    'custom_disruption', 
    create_custom_disruption
)
```

### Extension Development

#### Plugin Architecture

```python
# plugins/my_plugin.py
class MyPlugin:
    """Example plugin for extending DisruptSC functionality."""
    
    def __init__(self, model):
        self.model = model
        
    def on_simulation_start(self):
        """Called when simulation starts."""
        pass
        
    def on_time_step(self, t):
        """Called each time step."""
        pass
        
    def on_simulation_end(self):
        """Called when simulation ends."""
        pass

# Register plugin
def register_plugin(model):
    plugin = MyPlugin(model)
    model.register_plugin(plugin)
    return plugin
```

## IDE Configuration

### VS Code

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/tmp": true,
        "**/output": true
    }
}
```

### PyCharm

1. **Set interpreter** to conda environment or venv
2. **Configure code style** to use Black formatter
3. **Enable inspections** for Python and type hints
4. **Set test runner** to pytest
5. **Configure run configurations** for main scripts

## Debugging

### Common Issues

#### Import Errors

```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/disrupt-sc"

# Or use development installation
pip install -e .
```

#### Memory Issues

```python
# Monitor memory usage
import tracemalloc

tracemalloc.start()

# Run code
run_simulation()

# Check memory
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f} MB")
print(f"Peak: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

#### Performance Issues

1. **Enable caching** - Use `--cache` options
2. **Reduce model size** - Increase cutoff parameters
3. **Profile execution** - Identify bottlenecks
4. **Use smaller datasets** - Test with reduced scope

### Debugging Tools

#### Python Debugger

```python
import pdb

def problematic_function():
    x = calculate_something()
    pdb.set_trace()  # Breakpoint
    y = process_result(x)
    return y
```

#### Logging Configuration

```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Use in code
logger = logging.getLogger(__name__)
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
```

## Release Process

### Version Management

Version is managed in `disruptsc/_version.py`:

```python
# Update version
__version__ = "1.0.9"
```

### Release Checklist

1. **Update version number** in `_version.py`
2. **Update changelog** with new features and fixes
3. **Run full test suite** to ensure stability
4. **Update documentation** for any new features
5. **Create release tag** in git
6. **Build and test packages** before distribution

### Documentation Updates

```bash
# Build documentation locally
mkdocs serve

# Build for deployment
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

## Community

### Getting Help

- **GitHub Issues** - Report bugs and request features
- **GitHub Discussions** - Ask questions and share ideas
- **Documentation** - Comprehensive guides and reference
- **Code Examples** - Tutorial notebooks and scripts

### Contributing

We welcome contributions from the community:

1. **Bug reports** - Help identify and fix issues
2. **Feature requests** - Suggest improvements
3. **Code contributions** - Submit pull requests
4. **Documentation** - Improve guides and examples
5. **Testing** - Help validate releases

### Code of Conduct

Please follow our community guidelines:

- **Be respectful** - Treat all participants with respect
- **Be constructive** - Provide helpful feedback
- **Be collaborative** - Work together toward common goals
- **Be inclusive** - Welcome contributors from all backgrounds