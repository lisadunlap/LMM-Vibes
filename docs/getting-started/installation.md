# Installation

This guide will help you install LMM-Vibes and its dependencies.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation Options

### Option 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/LMM-Vibes.git
cd LMM-Vibes

# Install in development mode
pip install -e .
```

### Option 2: Install Dependencies Only

```bash
# Install required packages
pip install -r requirements.txt
```

## Verification

To verify your installation, run:

```bash
python -c "import lmmvibes; print('LMM-Vibes installed successfully!')"
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're in the correct directory and have installed the package
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Version Conflicts**: Consider using a virtual environment

### Using a Virtual Environment

```bash
# Create virtual environment
python -m venv lmmvibes-env

# Activate (Linux/Mac)
source lmmvibes-env/bin/activate

# Activate (Windows)
lmmvibes-env\Scripts\activate

# Install package
pip install -e .
```

## Next Steps

Once installed, check out the [Quick Start Guide](quick-start.md) to begin using LMM-Vibes. 