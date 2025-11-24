#!/bin/bash
set -e

ENV_NAME="nanochat-rl"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "conda could not be found. Please install conda first."
    exit 1
fi

# Create conda environment if it doesn't exist
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python=3.10 -y
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install Rust (needed for maturin)
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "Rust is already installed."
fi

# Install dependencies
echo "Installing dependencies..."
# Install build dependencies first
pip install maturin

# Install project dependencies
# We manually list them here based on pyproject.toml to avoid uv dependency if possible,
# but installing from . is easiest if we trust pip to handle pyproject.toml
pip install -e .

# Install dev dependencies
pip install pytest

echo "Setup complete! Activate the environment with: conda activate $ENV_NAME"
