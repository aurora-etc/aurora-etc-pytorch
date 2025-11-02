# Contributing to AURORA-ETC

Thank you for your interest in contributing to AURORA-ETC! This document provides guidelines for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/aurora-etc.git
   cd aurora-etc
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install package in editable mode
   ```

5. Install development dependencies:
   ```bash
   pip install pytest pytest-cov black flake8 mypy
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use `black` for code formatting:
  ```bash
  black aurora_etc/ scripts/ tests/
  ```
- Use `flake8` for linting:
  ```bash
  flake8 aurora_etc/ scripts/ tests/
  ```
- Type hints are encouraged where appropriate

## Testing

- Write unit tests for new features
- Run tests with:
  ```bash
  pytest tests/
  ```
- Aim for >80% code coverage

## Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git commit -m "Add: description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a Pull Request on GitHub

## Code Review

- All PRs require at least one approval
- Address review comments promptly
- Keep PRs focused and reasonably sized

## Documentation

- Update docstrings for new functions/classes
- Update README.md if adding new features
- Add examples for complex functionality

## Questions?

Open an issue for questions or discussion.

