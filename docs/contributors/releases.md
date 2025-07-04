# Release Management

This document explains how the automated release workflow works for DisruptSC.

## Overview

DisruptSC uses an automated release system that:

- **Frequent Development**: Push code changes frequently without version updates
- **Automated Releases**: When version is incremented, automatically create GitHub releases
- **Documentation Updates**: Version appears in MkDocs and README automatically
- **Version Management**: Single source of truth in `src/disruptsc/_version.py`

## Workflow

### Development (Frequent Pushes)

For regular development work:

1. **Make code changes** as usual
2. **Push to main branch** frequently
3. **Documentation deploys** automatically (latest version)
4. **No version increment** needed

```bash
# Regular development workflow
git add .
git commit -m "fix: improve error handling in transport network"
git push origin main

# Documentation automatically updates at https://ccolon.github.io/disrupt-sc/
```

### Creating a Release

When ready to create a formal release:

1. **Update version** in `src/disruptsc/_version.py`
2. **Optionally create custom release notes** (see Release Notes Control below)
3. **Push to main branch**
4. **Automation handles the rest**

```bash
# 1. Update version
echo '__version__ = "1.2.0"' > src/disruptsc/_version.py

# 2. Commit and push
git add src/disruptsc/_version.py
git commit -m "release: bump version to 1.2.0"
git push origin main

# 3. Automation creates:
# - GitHub release with tag v1.2.0
# - Updated documentation with version 1.2.0
# - Version badge in README.md
# - Release notes (auto-generated or custom)
```

## What Happens Automatically

When you increment the version number, the GitHub Actions workflow:

### 1. Detects Version Change
- Compares current version with previous commit
- Only triggers release workflow if version changed

### 2. Creates GitHub Release
- **Tag**: `v{version}` (e.g., `v1.2.0`)
- **Release notes**: Auto-generated from commit messages since last release
- **Installation instructions**: Updated with new version

### 3. Updates Documentation
- **MkDocs**: Deploys version-specific documentation
- **Version display**: Shows current version on documentation pages
- **Version selector**: Available in documentation sidebar

### 4. Updates Repository
- **README badge**: Version badge automatically updated
- **Git tags**: Version tag created and pushed

## Version Numbering

Use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes (`2.0.0`)
- **MINOR**: New functionality, backwards compatible (`1.1.0`)
- **PATCH**: Bug fixes, backwards compatible (`1.0.1`)

```python
# Examples of version updates
__version__ = "1.0.1"  # Bug fix
__version__ = "1.1.0"  # New feature
__version__ = "2.0.0"  # Breaking change
```

## Documentation Versioning

Documentation is managed with [mike](https://github.com/jimporter/mike):

- **Latest**: Always points to newest version
- **Version-specific**: Each release gets its own documentation
- **Navigation**: Version selector in documentation

### Available Documentation Versions

- **Latest**: `https://ccolon.github.io/disrupt-sc/` (default)
- **Specific**: `https://ccolon.github.io/disrupt-sc/1.2.0/`
- **Version list**: Available in documentation version selector

## Manual Operations

### Creating Pre-releases

For beta or release candidate versions:

```bash
# Update to pre-release version
echo '__version__ = "1.2.0-beta1"' > src/disruptsc/_version.py
git add src/disruptsc/_version.py
git commit -m "release: 1.2.0-beta1"
git push origin main

# Manually mark as pre-release in GitHub UI if needed
```

### Manual Documentation Deployment

If needed, you can manually deploy documentation:

```bash
# Install requirements
pip install -r docs-requirements.txt
pip install mike

# Deploy specific version
mike deploy --push --update-aliases 1.2.0 latest

# Set as default
mike set-default --push latest
```

### Emergency Release Fix

If automation fails:

```bash
# Create tag manually
git tag v1.2.0
git push origin v1.2.0

# Create release manually on GitHub
# Deploy docs manually (see above)
```

## Troubleshooting

### Release Workflow Not Triggered

**Problem**: Version changed but no release created

**Solution**: Check GitHub Actions logs
```bash
# Check if workflow ran
# Go to: https://github.com/ccolon/disrupt-sc/actions

# Common issues:
# - Workflow file syntax error
# - Permission issues
# - Version format invalid
```

### Documentation Not Updated

**Problem**: Documentation doesn't show new version

**Solution**: 
```bash
# Check docs-requirements.txt has mkdocs-macros-plugin
# Check docs/main.py exists and is correct
# Check MkDocs configuration includes macros plugin
```

### Version Badge Not Updated

**Problem**: README badge shows old version

**Solution**: Workflow should auto-update, but manual fix:
```bash
# Update badge manually
sed -i 's/version-[0-9.]*/version-1.2.0/' README.md
git add README.md
git commit -m "docs: update version badge"
git push origin main
```

## Release Notes Control

You have multiple options to control release notes, from fully automated to completely custom:

### Option 1: Automatic Release Notes (Default)

By default, release notes are auto-generated from commit messages with categorization:

- **✨ New Features**: Commits starting with `feat:`
- **🐛 Bug Fixes**: Commits starting with `fix:`
- **📚 Documentation**: Commits starting with `docs:`
- **🔧 Other Changes**: All other commits

### Option 2: Custom Release Notes (Pre-Release)

Create a `RELEASE_NOTES.md` file before incrementing the version:

```bash
# 1. Create custom release notes
cat > RELEASE_NOTES.md << 'EOF'
# Release Notes for v{{VERSION}}

## 🚀 Major Features

- **New simulation type**: Added sensitivity analysis with parameter sweeps
- **Memory management**: Resolved memory leaks in Monte Carlo simulations
- **Performance improvements**: 40% faster route calculations

## 🔧 Technical Changes

- Unified AdHocExecutor classes to reduce code duplication
- Enhanced input validation with comprehensive error reporting
- Updated documentation with new simulation types

## 🐛 Bug Fixes

- Fixed inventory parameter setting in sensitivity analysis
- Resolved transport network caching issues
- Corrected country loss calculation edge cases

## ⚠️ Breaking Changes

None in this release.

## 📦 Migration Guide

No migration needed - all changes are backwards compatible.
EOF

# 2. Update version and push together
echo '__version__ = "1.2.0"' > src/disruptsc/_version.py
git add RELEASE_NOTES.md src/disruptsc/_version.py
git commit -m "release: version 1.2.0 with custom release notes"
git push origin main

# 3. Automation uses your custom notes and cleans up the file
```

**Template placeholders:**
- `{{VERSION}}` - Automatically replaced with actual version number

### Option 3: Edit After Release (Manual)

Edit the release on GitHub after it's created:

1. **Release created automatically** with basic notes
2. **Go to GitHub**: `https://github.com/ccolon/disrupt-sc/releases`
3. **Click "Edit release"** on the new release
4. **Rewrite the description** as needed
5. **Save changes**

### Option 4: Release Drafts

Modify the workflow to create draft releases:

```yaml
# In .github/workflows/release.yml, change:
draft: true  # Instead of false
```

Then manually review and publish each release.

## Best Practices

### Commit Messages

Use conventional commits for better automatic release notes:

```bash
# Good commit messages for release notes
git commit -m "feat: add sensitivity analysis simulation type"
git commit -m "fix: resolve memory leak in monte carlo executor"  
git commit -m "docs: update parameter documentation"
git commit -m "refactor: simplify transport network setup"
```

### Custom Release Notes Template

Save this template for consistent custom release notes:

```markdown
# Release Notes for v{{VERSION}}

## 🚀 What's New

[Highlight major features and improvements]

## 🔧 Technical Changes

[Detail technical improvements, refactoring, etc.]

## 🐛 Bug Fixes

[List important bug fixes]

## ⚠️ Breaking Changes

[Note any breaking changes or "None in this release"]

## 📦 Migration Guide

[Provide migration steps or "No migration needed"]

## 🙏 Contributors

[Thank contributors if applicable]
```

### Release Timing

- **Patch releases**: As needed for bug fixes
- **Minor releases**: Monthly or when significant features ready
- **Major releases**: Quarterly or for breaking changes

### Version Strategy

```bash
# Development cycle example:
1.0.0 -> 1.0.1 -> 1.0.2 -> 1.1.0 -> 1.1.1 -> 2.0.0

# Pre-release testing:
1.1.0-beta1 -> 1.1.0-beta2 -> 1.1.0-rc1 -> 1.1.0
```

## Related Files

Key files in the release system:

- **`src/disruptsc/_version.py`** - Single source of truth for version
- **`.github/workflows/release.yml`** - Automated release workflow
- **`docs/main.py`** - MkDocs macros for version display
- **`docs-requirements.txt`** - Documentation dependencies
- **`mkdocs.yml`** - Documentation configuration with versioning
- **`README.md`** - Contains version badge

## Monitoring

Monitor the release system:

- **GitHub Actions**: Check workflow success/failure
- **Documentation**: Verify version appears correctly
- **Releases**: Check GitHub releases page
- **Tags**: Verify git tags created properly

The automation handles most release tasks, allowing you to focus on development while maintaining professional release management.