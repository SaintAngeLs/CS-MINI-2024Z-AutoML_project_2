#!/bin/bash

set -e 

echo "Starting deployment script..."

if ! git log -1 --pretty=%B | grep -q '\[deploy\]'; then
    echo "Predictively intentionally exiting deployment."
    exit 0
fi

if [ -f VERSION ]; then
    version=$(cat VERSION)
    IFS='.' read -r major minor patch <<<"$version"
    new_version="$major.$minor.$((patch + 1))"
    echo "$new_version" > VERSION
    echo "Updated version to $new_version"
else
    echo "0.1.0" > VERSION
    echo "VERSION file not found. Created with version 0.1.0"
fi

echo "Building package..."
python -m build --sdist --wheel --outdir dist/

echo "Uploading package to PyPI..."
python -m twine upload --non-interactive --repository-url https://upload.pypi.org/legacy/ dist/*

echo "Deployment completed successfully!"
