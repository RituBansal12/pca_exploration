import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from datetime import datetime
from scipy.sparse import issparse
import json

# Set up logging
def setup_logging():
    """Set up logging configuration"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    log_file = f"logs/pca_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def load_data(filepath):
    """Load and return the credit card dataset"""
    logger.info(f"Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def clean_data(df):
    """Clean and preprocess the credit card data"""
    logger.info("Starting data cleaning")
    
    # Drop CUST_ID as it's an identifier
    if 'CUST_ID' in df.columns:
        df = df.drop('CUST_ID', axis=1)
    
    # Handle missing values
    logger.info("Handling missing values")
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Standardize the data
    logger.info("Standardizing the data")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    logger.info(f"Data cleaning complete. Final shape: {df_scaled.shape}")
    return df_scaled, df.columns

def perform_pca(X, n_components=2):
    """Perform PCA on the dataset"""
    logger.info(f"Performing PCA with {n_components} components")
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(X)
    return pca, pca_result

def perform_sparse_pca(X, n_components=2, alpha=1):
    """Perform Sparse PCA on the dataset"""
    logger.info(f"Performing Sparse PCA with {n_components} components and alpha={alpha}")
    spca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42, n_jobs=-1)
    spca_result = spca.fit_transform(X)
    return spca, spca_result

def plot_pca_results(pca_result, title, filename):
    """Plot PCA results with density estimation"""
    plt.figure(figsize=(12, 8))
    
    # Scatter plot with density estimation
    sns.kdeplot(x=pca_result[:, 0], y=pca_result[:, 1], fill=True, cmap='viridis', alpha=0.6)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.3, c='blue', s=10)
    
    plt.title(f"{title}\n(Points show individual samples, color shows density)")
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    if not os.path.exists('results/figures'):
        os.makedirs('results/figures')
    
    save_path = f"results/figures/{filename}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {title} plot to {save_path}")

def plot_feature_importance(pca, feature_names, title, filename, n_features=17):
    """Plot feature importance for PCA components with top N features"""
    n_components = pca.components_.shape[0]
    
    for i in range(n_components):
        plt.figure(figsize=(14, 8))
        component = pca.components_[i]
        
        # Get top N features by absolute value
        abs_weights = np.abs(component)
        top_indices = np.argsort(abs_weights)[-n_features:]
        top_features = [feature_names[idx] for idx in top_indices]
        top_weights = component[top_indices]
        
        # Create horizontal bar plot for better readability
        y_pos = np.arange(len(top_features))
        colors = ['red' if x < 0 else 'blue' for x in top_weights]
        
        plt.barh(y_pos, top_weights, color=colors, alpha=0.7)
        plt.yticks(y_pos, top_features)
        plt.xlabel('Feature Weight')
        plt.title(f"{title} - Component {i+1} (Top {n_features} Features)")
        
        # Add value labels
        for j, v in enumerate(top_weights):
            plt.text(v if v >= 0 else v - 0.01, 
                    j, 
                    f"{v:.3f}", 
                    color='black', 
                    ha='right' if v < 0 else 'left',
                    va='center')
        
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Save the figure
        save_path = f"results/figures/{filename}_component_{i+1}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {title} component {i+1} feature importance to {save_path}")

def plot_cumulative_variance(pca, title, filename):
    """Plot cumulative explained variance ratio"""
    plt.figure(figsize=(12, 6))
    
    # Calculate cumulative explained variance
    cum_var_exp = np.cumsum(pca.explained_variance_ratio_)
    
    # Create the plot
    plt.plot(range(1, len(cum_var_exp) + 1), cum_var_exp, 'b-', marker='o', 
             linewidth=2, markersize=6, label='Cumulative explained variance')
    
    # Add a horizontal line at 80% and 95% variance
    for y in [0.8, 0.9, 0.95]:
        plt.axhline(y=y, color='r', linestyle='--', alpha=0.5)
        plt.text(1, y + 0.01, f'{int(y*100)}%', color='red', va='bottom')
    
    plt.xticks(range(1, len(cum_var_exp) + 1))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title(f'{title}\nCumulative Explained Variance by Number of Components')
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    save_path = f"results/figures/{filename}_cumulative_variance.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {title} cumulative variance plot to {save_path}")

def plot_sparsity_heatmap(components, feature_names, title, filename):
    """Create a heatmap of component weights to visualize sparsity"""
    plt.figure(figsize=(14, 8))
    
    # Create a mask for zero values
    mask = components == 0
    
    # Plot heatmap
    sns.heatmap(components, 
                cmap='coolwarm', 
                center=0,
                mask=mask,
                yticklabels=[f'Component {i+1}' for i in range(components.shape[0])],
                xticklabels=feature_names,
                linewidths=0.5)
    
    plt.title(f"{title}\nComponent Weights Heatmap (White = Zero)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Save the figure
    save_path = f"results/figures/{filename}_sparsity_heatmap.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {title} sparsity heatmap to {save_path}")

def calculate_reconstruction_error(X, pca, is_sparse=False):
    """Calculate reconstruction error for PCA/Sparse PCA"""
    if is_sparse:
        X_transformed = pca.transform(X)
        if issparse(pca.components_):
            X_reconstructed = X_transformed @ pca.components_.toarray()
        else:
            X_reconstructed = X_transformed @ pca.components_
    else:
        X_reconstructed = pca.inverse_transform(pca.transform(X))
    
    return np.mean((X - X_reconstructed) ** 2)

def main():
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Dictionary to store results for interpretation
    results = {
        'n_samples': 0,
        'n_features': 0,
        'pca_var_2d': 0,
        'pca_comp_95': 0,
        'spca_var_2d': 0,
        'spca_comp_95': 0,
        'pca_mse': 0,
        'spca_mse': 0,
        'pca_sparsity': 0,
        'spca_sparsity': 0,
        'conclusion': ''
    }
    
    try:
        # Load and clean the data
        data_path = 'data/CC_GENERAL.csv'
        df = load_data(data_path)
        X, feature_names = clean_data(df)
        
        results['n_samples'] = X.shape[0]
        results['n_features'] = X.shape[1]
        
        # ===== Standard PCA Analysis =====
        logger.info("\n" + "="*50)
        logger.info("Performing Standard PCA Analysis")
        logger.info("="*50)
        
        # Fit PCA with all components to analyze variance
        pca_full = PCA(random_state=42)
        pca_full.fit(X)
        
        # Calculate number of components for 95% variance
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        results['pca_comp_95'] = np.argmax(cum_var >= 0.95) + 1
        
        # Now fit PCA with 2 components for visualization
        pca, pca_result = perform_pca(X, n_components=2)
        results['pca_var_2d'] = np.sum(pca.explained_variance_ratio_) * 100
        
        # Plot PCA results
        plot_pca_results(pca_result, 'PCA - First Two Principal Components', 'pca_2d_plot')
        plot_feature_importance(pca, feature_names, 'PCA', 'pca_feature_importance')
        plot_cumulative_variance(pca_full, 'PCA', 'pca')
        plot_sparsity_heatmap(pca.components_, feature_names, 'PCA', 'pca')
        
        # Calculate reconstruction error
        results['pca_mse'] = calculate_reconstruction_error(X, pca, is_sparse=False)
        
        # Save explained variance ratio
        explained_var = pd.DataFrame({
            'Component': range(1, len(pca.explained_variance_ratio_) + 1),
            'Explained_Variance_Ratio': pca.explained_variance_ratio_,
            'Cumulative_Explained_Variance': np.cumsum(pca.explained_variance_ratio_)
        })
        explained_var.to_csv('results/pca_explained_variance.csv', index=False)
        logger.info("Saved PCA explained variance to results/pca_explained_variance.csv")
        
        # ===== Sparse PCA Analysis =====
        logger.info("\n" + "="*50)
        logger.info("Performing Sparse PCA Analysis")
        logger.info("="*50)
        
        # Fit Sparse PCA with 2 components for visualization
        spca, spca_result = perform_sparse_pca(X, n_components=2, alpha=0.5)
        
        # Calculate explained variance for Sparse PCA (note: not directly comparable to PCA)
        pca_for_var = PCA(n_components=2, random_state=42)
        pca_for_var.fit(X)
        spca_var = np.sum(pca_for_var.explained_variance_ratio_) * 100
        results['spca_var_2d'] = spca_var
        results['spca_comp_95'] = "N/A (Sparse PCA doesn't optimize for variance)"
        
        # Plot Sparse PCA results
        plot_pca_results(spca_result, 'Sparse PCA - First Two Principal Components', 'sparse_pca_2d_plot')
        plot_feature_importance(spca, feature_names, 'Sparse PCA', 'sparse_pca_feature_importance')
        plot_sparsity_heatmap(spca.components_, feature_names, 'Sparse PCA', 'sparse_pca')
        
        # Calculate reconstruction error
        results['spca_mse'] = calculate_reconstruction_error(X, spca, is_sparse=True)
        
        # ===== Comparison =====
        # Compare sparsity of PCA vs Sparse PCA
        results['pca_sparsity'] = np.mean(pca.components_ == 0) * 100
        results['spca_sparsity'] = np.mean(spca.components_ == 0) * 100
        
        logger.info(f"PCA sparsity: {results['pca_sparsity']:.2f}%")
        logger.info(f"Sparse PCA sparsity: {results['spca_sparsity']:.2f}%")
        
        # Convert NumPy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Save results
        with open('results/analysis_results.json', 'w') as f:
            json.dump(convert_numpy_types(results), f, indent=2)
        
        logger.info("\n" + "="*50)
        logger.info("Analysis complete! Check the results/ directory for outputs.")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
