import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import load_model_config, load_data_config, load_train_config
from model_training.loader import MetricsDataset
from train_model import get_data_paths
from matplotlib.scale import SymmetricalLogScale


ZOOM_PERCENTILE = 90

def plot_feature_distributions(dataset, features, devices, output_dir='feature_distributions'):
    """Plot distributions of all features after scaling."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate MDT and OST features
    mdt_data = dataset.mdt_features  # shape: (n_samples, n_mdts, n_features)
    ost_data = dataset.ost_features  # shape: (n_samples, n_osts, n_features)
    print(f"mdt_data shape: {mdt_data.shape}")
    print(f"ost_data shape: {ost_data.shape}")
    
    # Create feature names
    mdt_feature_names = []
    for feature in features['mdt_trace']:
        mdt_feature_names.append(feature)
    for feature in features['stats']:
        mdt_feature_names.append(feature)
     
    ost_feature_names = []
    for feature in features['ost_trace']:
        ost_feature_names.append(feature)
    for feature in features['stats']:
        ost_feature_names.append(feature)

    print(f"ost_feature_names: {ost_feature_names}")


    # Plot MDT feature distributions
    for i in range(mdt_data.shape[2]):  # iterate over features
        plt.figure(figsize=(15, 10))
        for j in range(mdt_data.shape[1]):  # iterate over MDTs
            feature_values = mdt_data[:, j, i]
            
            # Create subplot for each MDT
            plt.subplot(2, (mdt_data.shape[1] + 1) // 2, j + 1)
            
            # Basic statistics
            mean = np.mean(feature_values)
            std = np.std(feature_values)
            median = np.median(feature_values)
            sum = np.sum(feature_values)
            
            # Create histogram with KDE
            sns.histplot(data=feature_values, kde=True)
            plt.title(f'MDT {j} - {mdt_feature_names[i]}\n'
                     f'Mean: {mean:.2f}\nStd: {std:.2f}\nMedian: {median:.2f}\nSum: {sum:.2f}')
            plt.xlabel('Scaled Value')
            plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'mdt_feature_{i}_{mdt_feature_names[i]}.png'))
        plt.close()

    # Plot OST feature distributions
    for i in range(ost_data.shape[2]):  # iterate over features
        plt.figure(figsize=(20, 15))
        for j in range(ost_data.shape[1]):  # iterate over OSTs
            feature_values = ost_data[:, j, i]
            
            # Create subplot for each OST
            plt.subplot(4, 4, j + 1)
            
            # Basic statistics
            mean = np.mean(feature_values)
            std = np.std(feature_values)
            median = np.median(feature_values)
            sum = np.sum(feature_values)
            
            # Create histogram with KDE
            sns.histplot(data=feature_values, kde=True)
            plt.title(f'OST {j} - {ost_feature_names[i]}\n'
                     f'Mean: {mean:.2f}\nStd: {std:.2f}\nMedian: {median:.2f}\nSum: {sum:.2f}')
            plt.xlabel('Scaled Value')
            plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'ost_feature_{i}_{ost_feature_names[i]}.png'))
        plt.close()

    # Also plot overall distributions
    plot_overall_distributions(mdt_data, ost_data, mdt_feature_names, ost_feature_names, output_dir)

def plot_overall_distributions(mdt_data, ost_data, mdt_feature_names, ost_feature_names, output_dir):
    """Plot overall distributions combining all MDTs/OSTs for each feature."""
    os.makedirs(os.path.join(output_dir, 'overall'), exist_ok=True)
    
    # Plot overall MDT feature distributions
    for i in range(mdt_data.shape[2]):
        plt.figure(figsize=(10, 6))
        feature_values = mdt_data[:, :, i].flatten()
        
        # Basic statistics
        mean = np.mean(feature_values)
        std = np.std(feature_values)
        median = np.median(feature_values)
        sum = np.sum(feature_values)
        
        # Create histogram with KDE
        sns.histplot(data=feature_values, kde=True)
        plt.title(f'Overall MDT Distribution: {mdt_feature_names[i]}\n'
                 f'Mean: {mean:.2f}, Std: {std:.2f}, Median: {median:.2f}, Sum: {sum:.2f}')
        plt.xlabel('Scaled Value')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'overall', f'mdt_feature_{i}_{mdt_feature_names[i]}_overall.png'))
        plt.close()

    # Plot overall OST feature distributions
    for i in range(ost_data.shape[2]):
        plt.figure(figsize=(10, 6))
        feature_values = ost_data[:, :, i].flatten()
        
        # Basic statistics
        mean = np.mean(feature_values)
        std = np.std(feature_values)
        median = np.median(feature_values)
        sum = np.sum(feature_values)
        # Create histogram with KDE
        sns.histplot(data=feature_values, kde=True)
        plt.title(f'Overall OST Distribution: {ost_feature_names[i]}\n'
                 f'Mean: {mean:.2f}, Std: {std:.2f}, Median: {median:.2f}, Sum: {sum:.2f}')
        plt.xlabel('Scaled Value')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'overall', f'ost_feature_{i}_{ost_feature_names[i]}_overall.png'))
        plt.close()

def plot_correlation_matrices(dataset, features, devices, output_dir='feature_correlations'):
    """Plot correlation matrices for MDT and OST features."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare feature names
    mdt_feature_names = []
    for feature in features['mdt_trace']:
        mdt_feature_names.extend([f"{device}_{feature}" for device in devices['mdt']])
    for feature in features['stats']:
        mdt_feature_names.extend([f"{device}_{feature}" for device in devices['mdt']])
        
    ost_feature_names = []
    for feature in features['ost_trace']:
        ost_feature_names.extend([f"{device}_{feature}" for device in devices['ost']])
    for feature in features['stats']:
        ost_feature_names.extend([f"{device}_{feature}" for device in devices['ost']])

    # Calculate and plot MDT correlations
    mdt_data = dataset.mdt_features.reshape(-1, dataset.mdt_features.shape[2])
    mdt_corr = np.corrcoef(mdt_data.T)
    
    plt.figure(figsize=(15, 15))
    sns.heatmap(mdt_corr, xticklabels=mdt_feature_names, yticklabels=mdt_feature_names, 
                cmap='coolwarm', center=0, annot=True, fmt='.2f', 
                square=True, linewidths=0.5)
    plt.title('MDT Feature Correlations')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mdt_correlations.png'))
    plt.close()

    # Calculate and plot OST correlations
    ost_data = dataset.ost_features.reshape(-1, dataset.ost_features.shape[2])
    ost_corr = np.corrcoef(ost_data.T)
    
    plt.figure(figsize=(15, 15))
    sns.heatmap(ost_corr, xticklabels=ost_feature_names, yticklabels=ost_feature_names, 
                cmap='coolwarm', center=0, annot=True, fmt='.2f', 
                square=True, linewidths=0.5)
    plt.title('OST Feature Correlations')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ost_correlations.png'))
    plt.close()

def analyze_target_distribution(dataset, output_dir='target_analysis'):
    """Analyze and plot the distribution of target values."""
    os.makedirs(output_dir, exist_ok=True)
    
    targets = dataset.target
    
    plt.figure(figsize=(10, 6))
    if targets.shape[1] == 1:  # Binary classification
        sns.histplot(data=targets.flatten(), bins=2)
        plt.title('Target Distribution (Binary Classification)\n'
                 f'Positive samples: {np.sum(targets)}\n'
                 f'Negative samples: {len(targets) - np.sum(targets)}')
    else:  # Multi-class classification
        class_counts = np.sum(targets, axis=0)
        plt.bar(range(len(class_counts)), class_counts)
        plt.title('Target Distribution (Multi-class Classification)')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
    plt.close()

def plot_overlaid_distributions(train_dataset, test_dataset, features, devices, output_dir='feature_distribution_comparison'):
    """Plot overlaid distributions of train and test features after scaling, separated by labels."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate MDT and OST features for both datasets
    train_mdt_data = train_dataset.mdt_features  # shape: (n_samples, n_mdts, n_features)
    train_ost_data = train_dataset.ost_features  # shape: (n_samples, n_osts, n_features)
    train_labels = train_dataset.target.flatten()  # Assuming binary labels
    
    test_mdt_data = test_dataset.mdt_features
    test_ost_data = test_dataset.ost_features
    test_labels = test_dataset.target.flatten()  # Assuming binary labels
    
    # Create feature names
    mdt_feature_names = []
    for feature in features['mdt_trace']:
        mdt_feature_names.append(feature)
    for feature in features['stats']:
        mdt_feature_names.append(feature)
     
    ost_feature_names = []
    for feature in features['ost_trace']:
        ost_feature_names.append(feature)
    for feature in features['stats']:
        ost_feature_names.append(feature)

    # Print dataset sizes for reference
    print(f"Train dataset size: {len(train_dataset)} samples (Label 0: {np.sum(train_labels==0)}, Label 1: {np.sum(train_labels==1)})")
    print(f"Test dataset size: {len(test_dataset)} samples (Label 0: {np.sum(test_labels==0)}, Label 1: {np.sum(test_labels==1)})")

    # Define colors and labels for the four groups
    colors = ['blue', 'green', 'red', 'purple']
    group_labels = ['Train (Label 0)', 'Train (Label 1)', 'Test (Label 0)', 'Test (Label 1)']

    # Plot overall MDT feature distributions
    for i in range(train_mdt_data.shape[2]):
        # Get feature values for train and test, separated by labels
        train_values_0 = train_mdt_data[train_labels==0, :, i].flatten()
        train_values_1 = train_mdt_data[train_labels==1, :, i].flatten()
        test_values_0 = test_mdt_data[test_labels==0, :, i].flatten()
        test_values_1 = test_mdt_data[test_labels==1, :, i].flatten()
        
        # All values for statistics
        all_values = [train_values_0, train_values_1, test_values_0, test_values_1]
        all_values_flat = np.concatenate(all_values)
        
        # Calculate statistics for each group
        means = [np.mean(vals) if len(vals) > 0 else np.nan for vals in all_values]
        stds = [np.std(vals) if len(vals) > 0 else np.nan for vals in all_values]
        maxes = [np.max(vals) if len(vals) > 0 else np.nan for vals in all_values]
        q75s = [np.percentile(vals, 75) if len(vals) > 0 else np.nan for vals in all_values]
        
        # Check if distribution is long-tailed
        is_long_tailed = False
        for vals, q75, max_val in zip(all_values, q75s, maxes):
            if len(vals) > 0 and max_val > 5 * q75:
                is_long_tailed = True
                break
        
        # Create a figure with two subplots if long-tailed
        if is_long_tailed:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
            
            # Full distribution (possibly with log scale)
            for vals, color, label in zip(all_values, colors, group_labels):
                if len(vals) > 0:
                    sns.histplot(vals, kde=True, color=color, alpha=0.4, label=label, 
                                stat='density', ax=ax1)
            
            # Determine if log scale would be helpful
            overall_max = np.max(all_values_flat)
            overall_min = np.min(all_values_flat)
            overall_mean = np.mean(all_values_flat)
            
            # Check if data spans multiple orders of magnitude
            if overall_max / overall_mean > 10 or overall_mean / abs(overall_min) > 10:
                # Check if we have both positive and negative values
                if overall_min < 0 and overall_max > 0:
                    # Use symmetric log scale
                    ax1.set_xscale('symlog', linthresh=0.1)  # linthresh controls the linear region around zero
                    ax1.set_title(f'MDT Feature: {mdt_feature_names[i]} (Symmetric Log Scale)')
                else:
                    # Use regular log scale for all-positive or all-negative data
                    ax1.set_xscale('log')
                    ax1.set_title(f'MDT Feature: {mdt_feature_names[i]} (Log Scale)')
            else:
                ax1.set_title(f'MDT Feature: {mdt_feature_names[i]} (Full Range)')
            
            # Add statistics table below the plot
            stats_text = "Group | Count | Mean | Std | Max\n"
            stats_text += "--- | --- | --- | --- | ---\n"
            for j, (label, vals, mean, std, max_val) in enumerate(zip(group_labels, all_values, means, stds, maxes)):
                stats_text += f"{label} | {len(vals)} | {mean:.2f} | {std:.2f} | {max_val:.2f}\n"
            
            ax1.text(0.5, -0.15, stats_text, transform=ax1.transAxes, ha='center', 
                    va='top', fontsize=9, family='monospace')
            
            ax1.set_xlabel('Scaled Value')
            ax1.set_ylabel('Density')
            ax1.legend(loc='upper right')
            
            # Zoomed view using ZOOM_PERCENTILE instead of hardcoded 95
            q_zoom = [np.percentile(vals, ZOOM_PERCENTILE) if len(vals) > 0 else np.nan for vals in all_values]
            zoom_limit = np.nanmax(q_zoom)
            
            # Filter values for zoomed view
            zoomed_values = [vals[vals <= zoom_limit] if len(vals) > 0 else np.array([]) for vals in all_values]
            
            for vals, color, label in zip(zoomed_values, colors, group_labels):
                if len(vals) > 0:
                    sns.histplot(vals, kde=True, color=color, alpha=0.4, label=label, 
                                stat='density', ax=ax2)
            
            # Add percentage of samples included in zoomed view
            zoom_text = "Group | % Included\n"
            zoom_text += "--- | ---\n"
            for j, (label, orig, zoomed) in enumerate(zip(group_labels, all_values, zoomed_values)):
                if len(orig) > 0:
                    pct = len(zoomed) / len(orig) * 100
                    zoom_text += f"{label} | {pct:.1f}%\n"
                else:
                    zoom_text += f"{label} | N/A\n"
            
            ax2.set_title(f'MDT Feature: {mdt_feature_names[i]} (Zoomed View - {ZOOM_PERCENTILE}th Percentile)')
            ax2.text(0.5, -0.15, zoom_text, transform=ax2.transAxes, ha='center', 
                    va='top', fontsize=9, family='monospace')
            
            ax2.set_xlabel('Scaled Value')
            ax2.set_ylabel('Density')
            ax2.legend(loc='upper right')
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)  # Make room for the stats text
            plt.savefig(os.path.join(output_dir, f'mdt_feature_{i}_{mdt_feature_names[i]}_comparison.png'))
            plt.close()
            
        else:
            # Regular single plot for well-behaved distributions
            plt.figure(figsize=(14, 10))
            
            for vals, color, label in zip(all_values, colors, group_labels):
                if len(vals) > 0:
                    sns.histplot(vals, kde=True, color=color, alpha=0.4, label=label, stat='density')
            
            # Add statistics table below the plot
            stats_text = "Group | Count | Mean | Std | Max\n"
            stats_text += "--- | --- | --- | --- | ---\n"
            for j, (label, vals, mean, std, max_val) in enumerate(zip(group_labels, all_values, means, stds, maxes)):
                stats_text += f"{label} | {len(vals)} | {mean:.2f} | {std:.2f} | {max_val:.2f}\n"
            
            plt.title(f'MDT Feature: {mdt_feature_names[i]}')
            plt.figtext(0.5, 0.01, stats_text, ha='center', va='bottom', fontsize=9, family='monospace')
            
            plt.xlabel('Scaled Value')
            plt.ylabel('Density')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)  # Make room for the stats text
            plt.savefig(os.path.join(output_dir, f'mdt_feature_{i}_{mdt_feature_names[i]}_comparison.png'))
            plt.close()

    # Plot overall OST feature distributions
    for i in range(train_ost_data.shape[2]):
        # Get feature values for train and test, separated by labels
        train_values_0 = train_ost_data[train_labels==0, :, i].flatten()
        train_values_1 = train_ost_data[train_labels==1, :, i].flatten()
        test_values_0 = test_ost_data[test_labels==0, :, i].flatten()
        test_values_1 = test_ost_data[test_labels==1, :, i].flatten()
        
        # All values for statistics
        all_values = [train_values_0, train_values_1, test_values_0, test_values_1]
        all_values_flat = np.concatenate(all_values)
        
        # Calculate statistics for each group
        means = [np.mean(vals) if len(vals) > 0 else np.nan for vals in all_values]
        stds = [np.std(vals) if len(vals) > 0 else np.nan for vals in all_values]
        maxes = [np.max(vals) if len(vals) > 0 else np.nan for vals in all_values]
        q75s = [np.percentile(vals, 75) if len(vals) > 0 else np.nan for vals in all_values]
        
        # Check if distribution is long-tailed
        is_long_tailed = False
        for vals, q75, max_val in zip(all_values, q75s, maxes):
            if len(vals) > 0 and max_val > 5 * q75:
                is_long_tailed = True
                break
        
        # Create a figure with two subplots if long-tailed
        if is_long_tailed:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
            
            # Full distribution (possibly with log scale)
            for vals, color, label in zip(all_values, colors, group_labels):
                if len(vals) > 0:
                    sns.histplot(vals, kde=True, color=color, alpha=0.4, label=label, 
                                stat='density', ax=ax1)
            
            # Determine if log scale would be helpful
            overall_max = np.max(all_values_flat)
            overall_min = np.min(all_values_flat)
            overall_mean = np.mean(all_values_flat)
            
            # Check if data spans multiple orders of magnitude
            if overall_max / overall_mean > 10 or overall_mean / abs(overall_min) > 10:
                # Check if we have both positive and negative values
                if overall_min < 0 and overall_max > 0:
                    # Use symmetric log scale
                    ax1.set_xscale('symlog', linthresh=0.1)  # linthresh controls the linear region around zero
                    ax1.set_title(f'OST Feature: {ost_feature_names[i]} (Symmetric Log Scale)')
                else:
                    # Use regular log scale for all-positive or all-negative data
                    ax1.set_xscale('log')
                    ax1.set_title(f'OST Feature: {ost_feature_names[i]} (Log Scale)')
            else:
                ax1.set_title(f'OST Feature: {ost_feature_names[i]} (Full Range)')
            
            # Add statistics table below the plot
            stats_text = "Group | Count | Mean | Std | Max\n"
            stats_text += "--- | --- | --- | --- | ---\n"
            for j, (label, vals, mean, std, max_val) in enumerate(zip(group_labels, all_values, means, stds, maxes)):
                stats_text += f"{label} | {len(vals)} | {mean:.2f} | {std:.2f} | {max_val:.2f}\n"
            
            ax1.text(0.5, -0.15, stats_text, transform=ax1.transAxes, ha='center', 
                    va='top', fontsize=9, family='monospace')
            
            ax1.set_xlabel('Scaled Value')
            ax1.set_ylabel('Density')
            ax1.legend(loc='upper right')
            
            # Zoomed view using ZOOM_PERCENTILE instead of hardcoded 95
            q_zoom = [np.percentile(vals, ZOOM_PERCENTILE) if len(vals) > 0 else np.nan for vals in all_values]
            zoom_limit = np.nanmax(q_zoom)
            
            # Filter values for zoomed view
            zoomed_values = [vals[vals <= zoom_limit] if len(vals) > 0 else np.array([]) for vals in all_values]
            
            for vals, color, label in zip(zoomed_values, colors, group_labels):
                if len(vals) > 0:
                    sns.histplot(vals, kde=True, color=color, alpha=0.4, label=label, 
                                stat='density', ax=ax2)
            
            # Add percentage of samples included in zoomed view
            zoom_text = "Group | % Included\n"
            zoom_text += "--- | ---\n"
            for j, (label, orig, zoomed) in enumerate(zip(group_labels, all_values, zoomed_values)):
                if len(orig) > 0:
                    pct = len(zoomed) / len(orig) * 100
                    zoom_text += f"{label} | {pct:.1f}%\n"
                else:
                    zoom_text += f"{label} | N/A\n"
            
            ax2.set_title(f'OST Feature: {ost_feature_names[i]} (Zoomed View - {ZOOM_PERCENTILE}th Percentile)')
            ax2.text(0.5, -0.15, zoom_text, transform=ax2.transAxes, ha='center', 
                    va='top', fontsize=9, family='monospace')
            
            ax2.set_xlabel('Scaled Value')
            ax2.set_ylabel('Density')
            ax2.legend(loc='upper right')
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)  # Make room for the stats text
            plt.savefig(os.path.join(output_dir, f'ost_feature_{i}_{ost_feature_names[i]}_comparison.png'))
            plt.close()
            
        else:
            # Regular single plot for well-behaved distributions
            plt.figure(figsize=(14, 10))
            
            for vals, color, label in zip(all_values, colors, group_labels):
                if len(vals) > 0:
                    sns.histplot(vals, kde=True, color=color, alpha=0.4, label=label, stat='density')
            
            # Add statistics table below the plot
            stats_text = "Group | Count | Mean | Std | Max\n"
            stats_text += "--- | --- | --- | --- | ---\n"
            for j, (label, vals, mean, std, max_val) in enumerate(zip(group_labels, all_values, means, stds, maxes)):
                stats_text += f"{label} | {len(vals)} | {mean:.2f} | {std:.2f} | {max_val:.2f}\n"
            
            plt.title(f'OST Feature: {ost_feature_names[i]}')
            plt.figtext(0.5, 0.01, stats_text, ha='center', va='bottom', fontsize=9, family='monospace')
            
            plt.xlabel('Scaled Value')
            plt.ylabel('Density')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)  # Make room for the stats text
            plt.savefig(os.path.join(output_dir, f'ost_feature_{i}_{ost_feature_names[i]}_comparison.png'))
            plt.close()

def main():
    # Load configurations
    model_config = load_model_config()
    train_config = load_train_config()
    data_config = load_data_config()
    config = {
        "model_config": model_config,
        "train_config": train_config,
        "data_config": data_config
    }

    # Get data paths
    train_sample_paths, test_sample_paths = get_data_paths(config)

    # Create dataset
    train_dataset = MetricsDataset(
        train_sample_paths, 
        features=config['model_config']['features'],
        train=True,
        window_sizes=[1.0]
    )

    test_dataset = MetricsDataset(
        test_sample_paths,
        features=config['model_config']['features'],
        train=False,
        window_sizes=[1.0]
    )

    """
    # Create analysis directories
    base_output_dir = 'data_analysis/train'
    os.makedirs(base_output_dir, exist_ok=True)

    # Perform analysis
    print("Plotting feature distributions...")
    plot_feature_distributions(
        train_dataset, 
        config['model_config']['features'],
        train_dataset.devices,
        output_dir=os.path.join(base_output_dir, 'feature_distributions')
    )


    print("Analyzing target distribution...")
    analyze_target_distribution(
        train_dataset,
        output_dir=os.path.join(base_output_dir, 'target_analysis')
    )

    base_output_dir = 'data_analysis/test'
    os.makedirs(base_output_dir, exist_ok=True)


    print("Analyzing target distribution...")
    analyze_target_distribution(
        test_dataset,
        output_dir=os.path.join(base_output_dir, 'target_analysis')
    )

    print("Plotting feature distributions...")
    plot_feature_distributions(
        test_dataset, 
        config['model_config']['features'],
        test_dataset.devices,
        output_dir=os.path.join(base_output_dir, 'feature_distributions')
    )
    """

    # After creating both datasets and running individual analyses
    print("Plotting overlaid feature distributions...")
    plot_overlaid_distributions(
        train_dataset,
        test_dataset,
        config['model_config']['features'],
        train_dataset.devices,
        output_dir='data_analysis/comparison/feature_distributions'
    )

if __name__ == "__main__":
    main() 