#!/bin/bash

if [ ! -d "venv" ]; then
  echo "Creating a new virtual environment..."
  python3 -m venv venv
  if [ $? -eq 0 ]; then
    echo "Virtual environment created successfully."
  else
    echo "Error: Failed to create a virtual environment." >&2
    exit 1
  fi
else
  echo "Virtual environment already exists."
fi

echo "Activating the virtual environment..."
source venv/bin/activate

if [ $? -eq 0 ]; then
  echo "Virtual environment activated successfully."
else
  echo "Error: Failed to activate the virtual environment." >&2
  exit 1
fi

if [ -f "requirements.txt" ]; then
  echo "Installing dependencies from requirements.txt..."
  pip install -r requirements.txt
  if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully."
  else
    echo "Error: Failed to install dependencies." >&2
    exit 1
  fi
else
  echo "Error: requirements.txt file not found. Please ensure it's in the project root." >&2
  exit 1
fi