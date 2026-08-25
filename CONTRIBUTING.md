# Contributing to TUR-DPO

Thank you for your interest in contributing to TUR-DPO! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/abdulhadyabas2/TUR-DPO.git
   cd TUR-DPO
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```
4. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Creating a Branch

Create a feature branch from `main`:
```bash
git checkout -b feature/your-feature-name
```

### Code Style

We use standard Python conventions:
- Follow PEP 8 style guidelines
- Use type hints where possible
- Add docstrings for all public functions and classes
- Keep functions focused and modular

### Running Tests

Run the full test suite:
```bash
pytest tests/ -v
```

Run specific tests:
```bash
pytest tests/test_topology.py -v
pytest tests/test_loss.py::TestTURDPOLoss -v
```

With coverage:
```bash
pytest tests/ --cov=turdpo --cov-report=html
```

### Pre-commit Checks

Before committing, ensure:
1. All tests pass
2. Code is properly formatted
3. Type hints are correct (if using mypy)

```bash
# Format code
black turdpo/ tests/
isort turdpo/ tests/

# Check types (optional)
mypy turdpo/

# Run tests
pytest tests/ -v
```

## Submitting Changes

1. **Commit your changes** with clear commit messages:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub

### Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Include tests for new functionality
- Update documentation if needed
- Ensure CI passes

## Reporting Issues

When reporting issues, please include:
- Python version
- PyTorch version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on the code, not the person
- Help others learn

## Project Structure

```
turdpo/
├── turdpo/           # Main package
│   ├── __init__.py
│   ├── topology.py   # Graph extraction and scoring
│   ├── uncertainty.py# Uncertainty estimation
│   ├── rewards.py    # Shaped reward computation
│   ├── loss.py       # Loss functions
│   ├── trainer.py    # Training loop
│   ├── verifier.py   # Node verification
│   ├── calibration.py# Calibration metrics
│   ├── data.py       # Dataset utilities
│   └── utils.py      # Helper functions
├── tests/            # Test suite
├── examples/         # Example scripts
├── configs/          # Configuration files
└── train.py          # Main training script
```

## Adding New Features

### Adding a New Loss Function

1. Add the loss class to `turdpo/loss.py`
2. Export it in `turdpo/__init__.py`
3. Add tests in `tests/test_loss.py`
4. Update documentation

### Adding a New Uncertainty Method

1. Implement in `turdpo/uncertainty.py`
2. Ensure it follows the `UncertaintyEstimator` interface
3. Add comprehensive tests
4. Document the mathematical formulation

## Questions?

Feel free to open an issue for questions or discussions. We're happy to help!
