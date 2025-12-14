# PCA vs Sparse PCA: Results Interpretation

## Overview
This analysis compares Principal Component Analysis (PCA) and Sparse PCA on the credit card dataset. Below are the key findings and visualizations.

## 1. Data Summary
- **Number of samples**: 8950
- **Number of features**: 17

## 2. Dimensionality Reduction Performance

### 2.1 Explained Variance
- **PCA**:
  - Variance explained by first 2 components: 47.6%
  - Components needed for 95% variance: 12 (out of 17)
  
- **Sparse PCA**:
  - Variance explained by first 2 components: 47.6%
  - Components needed for 95% variance: N/A (Sparse PCA doesn't optimize for variance) (out of 17)

### 2.2 Reconstruction Error
- **PCA MSE**: 0.5239
- **Sparse PCA MSE**: 0.5240

## 3. Interpretability

### 3.1 Sparsity
- **PCA sparsity**: 0.0% of weights are exactly zero
- **Sparse PCA sparsity**: 5.9% of weights are exactly zero

### 3.2 Feature Importance
- PCA components typically use all features (dense representation)
- Sparse PCA components use only a subset of features (sparse representation)
- Check the heatmaps in the figures directory to visualize the sparsity patterns

## 4. When to Use Which?

### Use PCA when:
- You want to maximize the variance captured by each component
- You need orthogonal components
- Interpretability of individual features is not the primary concern
- You're using PCA as a preprocessing step for other algorithms

### Use Sparse PCA when:
- You need interpretable components with fewer non-zero weights
- Feature selection is important for your analysis
- You're working with high-dimensional data where many features may be irrelevant
- You want to identify the most important features in your dataset

## 5. Visualizations
See the following visualizations in the `results/figures/` directory:
1. `pca_2d_plot.png` - PCA projection of the first two components
2. `sparse_pca_2d_plot.png` - Sparse PCA projection of the first two components
3. `pca_cumulative_variance.png` - PCA cumulative explained variance
4. `sparse_pca_cumulative_variance.png` - Sparse PCA cumulative explained variance
5. `*_feature_importance_*.png` - Top features for each component
6. `*_sparsity_heatmap.png` - Heatmaps showing component sparsity patterns

## 6. Conclusion
Sparse PCA successfully achieved its goal of creating sparser components 
            compared to standard PCA. While it may capture slightly less variance with the same number of components, 
            the resulting components are more interpretable as they focus on fewer features. This makes Sparse PCA 
            particularly valuable when you need to identify which features are most important in your data.
