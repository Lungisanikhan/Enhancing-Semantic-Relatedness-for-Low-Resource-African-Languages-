# Semantic Relatedness for Low-Resource African Languages

This repository contains the implementation and results for the project:

**_Enhancing Semantic Relatedness for Low-Resource African Languages via Transfer Learning and M2M-100 Data Augmentation_**

The project investigates scalable and effective approaches for improving Semantic Textual Relatedness (STR) in low-resource African languages through transfer learning and multilingual machine translation–based augmentation.

##  Key Features

- Three African Languages: Hausa, Kinyarwanda, Afrikaans
- SemRel2024 Dataset: Standardized benchmark for semantic relatedness
- Transfer Learning: Fine-tuning African-centric models (AfriBERTa, AfroXLMR)
- M2M-100 Augmentation: MAFAND-MT back-translation pipeline
- Significant Gains: Up to **1167% improvement** over baseline models

## 📚 Research Questions

**RQ1:** Which transfer-learning methods (AfriBERTa vs AfroXLMR) yield the best STR performance for African languages?

**RQ2:** How effective is M2M-100 back-translation in improving model accuracy and robustness under low-resource constraints?

## 📊 Results Summary

| Language     | Baseline (XLM-R) | Fine-tuned (AfroXLMR) | M2M-100 Augmented | Improvement |
|--------------|------------------|-------------------------|-------------------|-------------|
| Afrikaans    | 0.4016           | 0.4085                  | 0.6452            | +60.66%     |
| Hausa        | 0.1247           | 0.6518                  | 0.6389            | +412.51%    |
| Kinyarwanda  | 0.0425           | 0.5390                  | 0.6456            | +1417.76%   |

**Metric:** Spearman Correlation Coefficient (ρ)

## 📁 Project Structure

```
├── COS802_Project_Code.ipynb
├── COS_802_Proposal_Final_u25743695.pdf
├── IEEE_Paper.tex
├── README.md
│
├── results/
│   ├── all_results_mafand.csv
│   ├── statistical_tests_mafand.csv
│   └── final_comparison_table_mafand.csv
│
├── visualizations/
│   ├── comparison_bar_chart_mafand.html
│   ├── improvement_heatmap_mafand.html
│   ├── progression_line_chart_mafand.html
│   └── predictions_scatter_mafand.html
│
├── augmented_data/
│   ├── afr_mafand_m2m100_augmented_train.csv
│   ├── hau_mafand_m2m100_augmented_train.csv
│   └── kin_mafand_m2m100_augmented_train.csv
│
└── models/
```

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/african-semantic-relatedness.git
cd african-semantic-relatedness
```

### 2. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets pandas numpy scipy scikit-learn
pip install sentencepiece sacremoses accelerate
pip install matplotlib seaborn plotly
```

## 🚀 Usage

### Quick Start
```python
from datasets import load_dataset
dataset = load_dataset("SemRel/SemRel2024", "hau")

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-base")
model = AutoModel.from_pretrained("path/to/finetuned/model")
```

## 🔧 Running the Full Pipeline

### Google Colab (Recommended)
1. Upload notebook  
2. Mount Drive  
3. Run all cells  
4. Outputs saved to: `/content/drive/MyDrive/COS802_NLP_Project/`

## 📘 Methodology (Summary)

- Dataset: Afrikaans, Hausa, Kinyarwanda  
- Baselines: mBERT, XLM-R  
- Transfer Learning: AfriBERTa, AfroXLMR  
- M2M-100 Augmentation: Back-translation (50%)  
- Evaluation: Spearman ρ + paired t-tests  

## 📊 Visualizations

Available in `visualizations/`:
- Bar charts  
- Heatmaps  
- Line charts  
- Scatter plots  

## 📄 License

MIT License

## 📬 Contact

**Lungisani Khanyile**  
u25743695@up.ac.za  
University of Pretoria  
COS802 – NLP
