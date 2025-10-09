# Tests

This directory contains unit tests for the handwritten math solver application.

## Running Tests

To run all tests, use pytest from the project root directory:

```bash
pytest
```

Or if using a virtual environment:

```bash
venv/Scripts/python.exe -m pytest
```

### Run specific test file:
```bash
pytest tests/test_solve_equation.py
```

### Run with verbose output:
```bash
pytest -v
```

### Run specific test:
```bash
pytest tests/test_solve_equation.py::TestSolveEquation::test_simple_equation_positive_result
```
