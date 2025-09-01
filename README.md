# BPMDR (Blocking Peptidoglycan of Multi-drug Resistance Pathogens)

A machine learning-powered tool for predicting biological activity (pIC50) of compounds against peptidoglycan biosynthesis in multidrug-resistant bacterial pathogens.

## 🎯 Overview

BPMDR accelerates the drug discovery process by predicting the biological activity of compounds (natural products, synthetic compounds, or existing drugs) against peptidoglycan biosynthesis pathways in MDR pathogens. This tool enables rapid virtual screening and prioritization of compounds for further experimental validation.

## 🚀 Key Applications

* **Virtual Screening**: Rapidly screen large compound libraries
* **Lead Optimization**: Predict activity of chemical modifications  
* **Natural Product Discovery**: Evaluate natural compounds for antibacterial activity
* **Drug Repurposing**: Identify existing drugs with potential anti-MDR activity
* **Decision Support**: Prioritize compounds for in-vitro/in-vivo validation

## 📊 Model Performance

Our validated machine learning models demonstrate excellent predictive performance:

| Model | Training Set (R²) | Validation Set (R²) | Test Set (R²) |
|-------|------------------|-------------------|---------------|
| Random Forest | 0.9467 | 0.9334 | 0.9573 |
| Gradient Boosting | 0.9583 | 0.9279 | 0.9551 |

## 🛠️ Installation

### Prerequisites

* Python 3.8+
* pip or conda package manager

### Quick Start

```bash
git clone https://github.com/yourusername/BPMDR.git
cd BPMDR
pip install -r requirements.txt
```

## 🏃‍♂️ Simple Start

### Command Line Usage

```bash
python predict.py
# Provide input file name (e.g. sample.csv) containing 2 columns: SMILES, pIC50 (Keep pIC50 Column Enpty) 
```

## 📝 Input Format

Your input CSV file should contain at minimum:

### Required Columns:
* **SMILES**: Simplified Molecular Input Line Entry System notation
* **pIC50**: Predicted Biological Activity (Keep the column empty)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use BPMDR in your research, please cite:

## 🔬 Scientific Background

Peptidoglycan biosynthesis is a critical pathway in bacterial cell wall formation and represents an attractive target for antibacterial drug development. Our machine learning models are trained on experimental data from various MDR pathogens including:

- *Staphylococcus aureus* (MRSA)
- *haemophilus influenzae* 
- *Pseudomonas aeruginosa*
- *Acinetobacter baumannii*

## ⚠️ Disclaimer

BPMDR is intended for research purposes only. Predictions should be validated experimentally before any clinical or commercial applications.

## 🙏 Acknowledgments

- Training data sourced from ChEMBL and literature curation
- Molecular descriptors computed using RDKit
- Special thanks to the open-source cheminformatics community

## 🙏 Contact

K.M. Tanjida Islam
Email: 

---
