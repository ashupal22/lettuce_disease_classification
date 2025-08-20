# 🚀 Quick Start Guide

## 1. Download Dataset
```bash
# Option 1: Using Kaggle CLI (requires account)
pip install kaggle
kaggle datasets download -d ashishjstar/lettuce-diseases
unzip lettuce-diseases.zip -d data/raw/

# Option 2: Manual download
# Visit: https://www.kaggle.com/datasets/ashishjstar/lettuce-diseases
# Download and extract to data/raw/Lettuce_disease_datasets/
```

## 2. Explore Dataset
```bash
python main.py --mode explore --data-dir data/raw/Lettuce_disease_datasets
```

## 3. Run Complete Pipeline
```bash
python main.py --mode full --data-dir data/raw/Lettuce_disease_datasets
```

## 4. Individual Approaches
```bash
# Classical ML only
python main.py --mode classical --data-dir data/raw/Lettuce_disease_datasets

# Deep Learning only
python main.py --mode deep_learning --data-dir data/raw/Lettuce_disease_datasets
```

## 5. Custom Configuration
```bash
python main.py --config config/sample_config.json --data-dir data/raw/Lettuce_disease_datasets
```

## 6. Check Results
- Plots: `results/plots/`
- Reports: `results/reports/`
- Models: `models/`
- Logs: `results/logs/`
