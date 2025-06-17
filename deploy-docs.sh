#!/bin/bash

# Deploy documentation to GitHub Pages
# This script builds the docs and deploys them using mkdocs gh-deploy

set -e  # Exit on any error

echo "🚀 Deploying DisruptSC Documentation to GitHub Pages"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo "❌ Error: mkdocs not found. Installing dependencies..."
    pip install -r docs-requirements.txt
    pip install mkdocs-mermaid2-plugin
fi

# Verify mkdocs.yml exists
if [ ! -f "mkdocs.yml" ]; then
    echo "❌ Error: mkdocs.yml not found"
    exit 1
fi

# Build and validate the documentation
echo "🔨 Building documentation..."
mkdocs build --clean --strict

if [ $? -ne 0 ]; then
    echo "❌ Error: Documentation build failed"
    exit 1
fi

echo "✅ Documentation built successfully"

# Deploy to GitHub Pages
echo "📤 Deploying to GitHub Pages..."
mkdocs gh-deploy --clean

if [ $? -eq 0 ]; then
    echo "✅ Documentation deployed successfully!"
    echo "📖 Your documentation will be available at: https://$(git config --get remote.origin.url | sed 's/.*github.com[:/]\([^/]*\)\/\([^/]*\)\.git/\1.github.io\/\2/')"
else
    echo "❌ Error: Deployment failed"
    exit 1
fi