# Changes

## 2.1.0

- Migrated FFX to uv / ruff and removed Travis CI for Github Actions

## 2.0.1 / 2.0.2

- Fix ImportError introduced in 2.0.0

## 2.0.0

- Added a Makefile
- Substantially refactored FFX to modernize the codebase. Broke up `core.py` into a module.
- Replaced custom CLI tooling with Click
- Adopted pytest for tests; improved test coverage
- Updated Travis CI to use a modern Python build matrix
- Added pylint, isort, black. See the `make validate` command
