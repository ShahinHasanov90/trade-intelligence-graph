# Contributing to Trade Intelligence Graph

Thank you for your interest in contributing to Trade Intelligence Graph. This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
3. Install development dependencies: `make install`
4. Run tests to verify setup: `make test`

## Development Workflow

1. Create a feature branch from `main`: `git checkout -b feature/your-feature`
2. Make your changes with appropriate tests
3. Run the full test suite: `make test`
4. Run linters: `make lint`
5. Format code: `make format`
6. Commit with a clear message
7. Open a pull request

## Code Standards

- **Type hints:** All functions must have complete type annotations
- **Docstrings:** All public functions and classes must have Google-style docstrings
- **Tests:** All new features must include tests (minimum 80% coverage)
- **Formatting:** Code is formatted with Black (line length 88)
- **Linting:** Code must pass ruff and mypy checks

## Architecture Decisions

Significant architectural changes should be proposed as Architecture Decision Records (ADRs) in `docs/adr/`. Use the existing ADRs as a template.

## Pull Request Process

1. Update documentation if your change affects the public API
2. Add tests for new functionality
3. Ensure all CI checks pass
4. Request review from a maintainer

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- For security vulnerabilities, see SECURITY.md
