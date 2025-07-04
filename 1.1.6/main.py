"""
MkDocs macros to inject version and other dynamic content.
"""

import os
import sys
from pathlib import Path

def define_env(env):
    """
    Define macros for MkDocs.
    """
    # Try multiple ways to get the version
    version_value = "1.1.5"  # Fallback default
    
    try:
        # Method 1: Try to import from source
        current_dir = Path(__file__).parent
        repo_root = current_dir.parent
        src_dir = repo_root / "src"
        
        if src_dir.exists():
            sys.path.insert(0, str(src_dir))
            from disruptsc._version import __version__
            version_value = __version__
    except Exception:
        try:
            # Method 2: Try to read the version file directly
            version_file = Path(__file__).parent.parent / "src" / "disruptsc" / "_version.py"
            if version_file.exists():
                version_content = version_file.read_text()
                for line in version_content.split('\n'):
                    if line.strip().startswith('__version__'):
                        version_value = line.split('=')[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass  # Use fallback
    
    env.variables['version'] = version_value
    
    @env.macro
    def version():
        """Return the current version."""
        return env.variables['version']
    
    @env.macro
    def version_badge():
        """Return a version badge for the current version."""
        v = env.variables['version']
        return f"[![Version](https://img.shields.io/badge/version-{v}-blue)](https://github.com/ccolon/disrupt-sc/releases/tag/v{v})"
    
    @env.macro
    def installation_instructions():
        """Return installation instructions with current version."""
        v = env.variables['version']
        return f"""```bash
# Clone the repository
git clone https://github.com/ccolon/disrupt-sc.git
cd disrupt-sc

# Checkout specific version (optional)
git checkout v{v}

# Create environment
conda env create -f dsc-environment.yml
conda activate dsc
```"""