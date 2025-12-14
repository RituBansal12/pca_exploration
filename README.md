# PCA vs Sparse PCA: A Comparative Analysis

This project provides a comprehensive comparison between Principal Component Analysis (PCA) and Sparse PCA using credit card customer data. The analysis includes data preprocessing, dimensionality reduction, visualization, and interpretation of results.

## Article 
https://medium.com/@ritu.bansalrb00/pca-vs-sparse-pca-a-practical-guide-to-interpretable-dimensionality-reduction-775760461ad0

## Key Features

- **Data Preprocessing**: Automatic handling of missing values and standardization
- **PCA Analysis**: Standard PCA with variance analysis
- **Sparse PCA**: Implementation with sparsity constraints
- **Comparative Analysis**: Direct comparison of PCA and Sparse PCA results
- **Visualizations**: Multiple plots for better understanding of the results
- **Detailed Reporting**: Comprehensive markdown report with interpretations

## Dataset

The dataset (`CC_GENERAL.csv`) contains credit card customer information with the following key features:
- `BALANCE`: Balance amount left in the account
- `PURCHASES`: Total purchase amount
- `CASH_ADVANCE`: Cash in advance given by the user
- `CREDIT_LIMIT`: Credit limit of the user
- `PAYMENTS`: Payment done by the user
- And several other spending-related features

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the analysis script:
```bash
python pca_analysis.py
```

## Results

The analysis generates the following outputs:

### Visualizations (`results/figures/`)
- `pca_2d_plot.png`: 2D projection using standard PCA
- `sparse_pca_2d_plot.png`: 2D projection using Sparse PCA
- `pca_feature_importance_*.png`: Top features for each PCA component
- `sparse_pca_feature_importance_*.png`: Top features for each Sparse PCA component
- `pca_cumulative_variance.png`: Explained variance by number of components
- `*_sparsity_heatmap.png`: Heatmaps showing component sparsity patterns

### Data Files (`results/`)
- `pca_explained_variance.csv`: Detailed variance explained by each component
- `analysis_results.json`: Complete analysis results in JSON format

## Project Structure

```
.
├── data/                   # Contains the input dataset
│   └── CC_GENERAL.csv
├── results/                # Analysis outputs
│   ├── figures/            # Generated visualizations
│   ├── analysis_results.json
│   └── pca_explained_variance.csv
├── logs/                   # Execution logs
├── pca_analysis.py         # Main analysis script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Dependencies

- Python 3.7+
- numpy >= 1.19.0
- pandas >= 1.0.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

## Interpreting Results

1. **PCA vs Sparse PCA**: Compare the sparsity patterns in the heatmaps
2. **Feature Importance**: Check which features contribute most to each component
3. **Variance Explained**: See how many components are needed to explain most of the variance
4. **Reconstruction Error**: Compare how well each method preserves the original data

Detailed interpretation is in `results_interpretation.md`
