# CLAUDE.md - SegmentIt Reference Guide

## Build & Development Commands
- `python -m pip install -e .` - Install package in development mode
- `python -m pytest tests/` - Run all tests
- `python -m pytest tests/test_file.py::test_name` - Run a single test
- `python -m black .` - Format code
- `python -m pylint segmentit` - Run linting
- `python -m mypy segmentit` - Run type checking
- `python scripts/train.py --config configs/default.yaml` - Train model
- `python scripts/evaluate.py --model_path models/model.onnx` - Evaluate model

## Code Style Guidelines
- Use Python type hints throughout
- Follow PEP 8 style guidelines
- Format code using Black
- Organize imports: std lib, third-party, local
- Use snake_case for functions/variables, PascalCase for classes
- Write docstrings in NumPy/Google style
- Handle exceptions with specific error types
- Use pathlib for file operations, not os.path
- Use dataclasses or named tuples for data structures