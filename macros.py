"""
MkDocs macros to inject version and other dynamic content.
"""

import os
import sys
from pathlib import Path

def define_env(env):
    """
    Define variables and macros for MkDocs.
    """
    # Try multiple ways to get the version
    version_value = "1.1.6"  # Fallback default
    
    try:
        # Method 1: Try to import from source
        current_dir = Path(__file__).parent
        src_dir = current_dir / "src"
        
        if src_dir.exists():
            sys.path.insert(0, str(src_dir))
            from disruptsc._version import __version__
            version_value = __version__
    except Exception:
        try:
            # Method 2: Try to read the version file directly
            version_file = Path(__file__).parent / "src" / "disruptsc" / "_version.py"
            if version_file.exists():
                version_content = version_file.read_text()
                for line in version_content.split('\n'):
                    if line.strip().startswith('__version__'):
                        version_value = line.split('=')[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass  # Use fallback
    
    # Set as simple variable instead of function
    env.variables['version'] = version_value
    
    # Also provide as macros for backwards compatibility
    @env.macro
    def get_version():
        """Return the current version."""
        return version_value
    
    @env.macro  
    def version_badge():
        """Return a version badge for the current version."""
        return f"[![Version](https://img.shields.io/badge/version-{version_value}-blue)](https://github.com/ccolon/disrupt-sc/releases/tag/v{version_value})"
    
    @env.macro
    def installation_instructions():
        """Return installation instructions with current version."""
        return f"""```bash
# Clone the repository
git clone https://github.com/ccolon/disrupt-sc.git
cd disrupt-sc

# Checkout specific version (optional)
git checkout v{version_value}

# Create environment
conda env create -f dsc-environment.yml
conda activate dsc
```"""