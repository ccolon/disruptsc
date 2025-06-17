# Documentation Deployment Guide

This guide explains how to deploy the DisruptSC documentation to GitHub Pages using automated or manual methods.

## Prerequisites

- Git repository hosted on GitHub
- MkDocs and dependencies installed
- Push access to the repository

## Method 1: Automated Deployment (Recommended)

### GitHub Actions Setup

The repository includes a GitHub Actions workflow (`.github/workflows/docs.yml`) that automatically builds and deploys documentation when changes are pushed to the main branch.

#### Step 1: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** tab
3. Scroll down to **Pages** section
4. Under **Source**, select **GitHub Actions**
5. Click **Save**

#### Step 2: Configure Repository Permissions

Ensure the workflow has proper permissions:

1. Go to **Settings** → **Actions** → **General**
2. Under **Workflow permissions**, select **Read and write permissions**
3. Check **Allow GitHub Actions to create and approve pull requests**
4. Click **Save**

#### Step 3: Trigger Deployment

The workflow automatically triggers when:
- Changes are pushed to the `main` branch
- Files in `docs/`, `mkdocs.yml`, or the workflow file are modified

```bash
# Make changes to documentation
git add docs/
git commit -m "Update documentation"
git push origin main
```

#### Step 4: Verify Deployment

1. Go to **Actions** tab in your repository
2. Check the **Deploy Documentation** workflow
3. Once complete, visit: `https://yourusername.github.io/disrupt-sc`

### Workflow Features

- **Automatic building** - Builds documentation on every push
- **Dependency caching** - Speeds up builds
- **Build validation** - Fails if documentation has errors
- **Branch protection** - Only deploys from main branch
- **Artifact upload** - Stores build artifacts

## Method 2: Manual Deployment

### Using the Deployment Script

Use the provided script for quick manual deployment:

```bash
# Make script executable (if not already)
chmod +x deploy-docs.sh

# Deploy documentation
./deploy-docs.sh
```

The script will:
1. Verify prerequisites
2. Build documentation
3. Deploy to GitHub Pages
4. Provide the documentation URL

### Manual MkDocs Deployment

For direct control over the deployment process:

#### Step 1: Install Dependencies

```bash
# Install documentation dependencies
pip install -r docs-requirements.txt
pip install mkdocs-mermaid2-plugin
```

#### Step 2: Build Documentation

```bash
# Build and validate documentation
mkdocs build --clean --strict
```

#### Step 3: Deploy to GitHub Pages

```bash
# Deploy to gh-pages branch
mkdocs gh-deploy --clean
```

#### Step 4: Verify Deployment

Visit your documentation site at:
`https://yourusername.github.io/repository-name`

## Configuration

### MkDocs Configuration

The documentation is configured in `mkdocs.yml`:

```yaml
site_name: DisruptSC Documentation
site_url: https://yourusername.github.io/disrupt-sc
repo_url: https://github.com/yourusername/disrupt-sc
repo_name: yourusername/disrupt-sc

# GitHub Pages configuration
site_dir: site
```

### Custom Domain (Optional)

To use a custom domain:

1. Add `CNAME` file to `docs/` directory:
   ```
   your-domain.com
   ```

2. Configure DNS with your domain provider:
   ```
   CNAME  docs.your-domain.com  yourusername.github.io
   ```

3. Update `mkdocs.yml`:
   ```yaml
   site_url: https://docs.your-domain.com
   ```

## Troubleshooting

### Common Issues

#### Build Failures

**Error**: `Config value: 'theme.features'`
**Solution**: Update mkdocs-material to latest version

**Error**: `Plugin 'mermaid2' not found`
**Solution**: Install missing plugin:
```bash
pip install mkdocs-mermaid2-plugin
```

#### Deployment Failures

**Error**: `Permission denied`
**Solution**: Check repository permissions and GitHub Actions settings

**Error**: `gh-pages branch not found`
**Solution**: The branch is created automatically on first deployment

#### Site Not Loading

**Issue**: 404 error on GitHub Pages URL
**Solutions**:
1. Check that GitHub Pages is enabled
2. Verify the source is set to "GitHub Actions"
3. Wait a few minutes for DNS propagation
4. Clear browser cache

### Debug Mode

Enable verbose output for troubleshooting:

```bash
# Build with verbose output
mkdocs build --verbose

# Deploy with verbose output
mkdocs gh-deploy --verbose
```

### Checking Deployment Status

Monitor deployment progress:

```bash
# Check git status
git status

# View recent commits
git log --oneline -5

# Check remote branches
git branch -r
```

## Advanced Configuration

### Branch-specific Deployment

Deploy from different branches:

```bash
# Deploy from current branch
mkdocs gh-deploy --clean

# Deploy specific branch
git checkout feature-docs
mkdocs gh-deploy --clean
```

### Version Management

Use `mike` for version management:

```bash
# Install mike
pip install mike

# Deploy specific version
mike deploy --push --update-aliases 1.0 latest

# List versions
mike list

# Set default version
mike set-default --push latest
```

### Build Optimization

Optimize build performance:

```yaml
# mkdocs.yml
plugins:
  - search
  - mermaid2:
      version: 10.0.2
  - section-index

# Exclude unnecessary files
watch:
  - docs
  - mkdocs.yml

# Use strict mode
strict: true
```

## Security Considerations

### Repository Access

- Limit write access to main branch
- Require pull request reviews
- Use branch protection rules

### Deployment Secrets

- Don't commit sensitive information
- Use GitHub Secrets for API keys
- Review workflow permissions regularly

### Content Security

- Validate all markdown content
- Check external links regularly
- Monitor for broken references

## Monitoring and Maintenance

### Regular Tasks

1. **Update dependencies** monthly:
   ```bash
   pip install --upgrade -r docs-requirements.txt
   ```

2. **Check build status** weekly:
   - Review GitHub Actions results
   - Test documentation links
   - Verify mobile responsiveness

3. **Content review** quarterly:
   - Update outdated information
   - Add new features documentation
   - Improve navigation structure

### Analytics (Optional)

Add Google Analytics to track usage:

```yaml
# mkdocs.yml
google_analytics:
  - UA-XXXXXXXX-X
  - auto
```

### Performance Monitoring

Monitor site performance:
- Use Lighthouse for performance audits
- Check loading times regularly
- Optimize images and assets

## Support

### Getting Help

- **GitHub Issues** - Report deployment problems
- **MkDocs Documentation** - https://www.mkdocs.org/
- **Material Theme Docs** - https://squidfunk.github.io/mkdocs-material/
- **GitHub Pages Docs** - https://docs.github.com/en/pages

### Community Resources

- MkDocs Community Forum
- GitHub Pages Community
- Stack Overflow (use tags: mkdocs, github-pages)

## Examples

### Complete Deployment Workflow

```bash
# 1. Make documentation changes
vim docs/user-guide/new-feature.md

# 2. Test locally
mkdocs serve

# 3. Build and validate
mkdocs build --strict

# 4. Commit changes
git add docs/
git commit -m "docs: add new feature documentation"

# 5. Push to trigger auto-deployment
git push origin main

# 6. Verify deployment
# Visit https://yourusername.github.io/disrupt-sc
```

### Emergency Manual Deployment

```bash
# If automated deployment fails
./deploy-docs.sh

# Or manually
mkdocs gh-deploy --force
```

This comprehensive deployment setup ensures your DisruptSC documentation is always up-to-date and accessible to users worldwide.