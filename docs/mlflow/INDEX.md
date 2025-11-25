# MLflow Documentation Index

This directory contains all MLflow configuration documentation for the Demand Forecasting project.

## 📚 Documentation Files

### Quick Start
- **[README.md](README.md)** - Start here! Quick reference guide with 3-step setup

### Detailed Guides
- **[SETUP.md](SETUP.md)** - Comprehensive setup and troubleshooting guide
- **[MIGRATION_EXAMPLES.md](MIGRATION_EXAMPLES.md)** - Before/after examples for updating notebooks

### Reference
- **[CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)** - Complete technical summary of all changes
- **[MESSAGE_FOR_OTHER_CHAT.md](MESSAGE_FOR_OTHER_CHAT.md)** - Template message for sharing changes with other Claude instances
- **[SUMMARY.txt](SUMMARY.txt)** - Visual summary with ASCII art

## 🎯 Quick Navigation

### I want to...

**Get started quickly**
→ Read [README.md](README.md)

**Update an existing notebook**
→ See [MIGRATION_EXAMPLES.md](MIGRATION_EXAMPLES.md)

**Troubleshoot an issue**
→ Check [SETUP.md](SETUP.md#troubleshooting)

**Understand what changed**
→ Read [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)

**Share changes with another Claude chat**
→ Use [MESSAGE_FOR_OTHER_CHAT.md](MESSAGE_FOR_OTHER_CHAT.md)

## 🚀 Essential Commands

### In Notebooks
```python
from config.mlflow_config import setup_mlflow
setup_mlflow("experiment_name")
```

### Start MLflow UI
```bash
./scripts/start_mlflow_ui.sh
```

### Access UI
http://127.0.0.1:5000

## 📂 Related Files

Outside this directory:
- `../../config/mlflow_config.py` - Centralized configuration module
- `../../scripts/start_mlflow_ui.sh` - UI startup script

## 🎉 What's Different Now?

**Before:** Mixed tracking directories, manual setup in each notebook
**After:** One location (`mlflow_results/`), one config module, one startup script

All your models in one place, always! 🎯
