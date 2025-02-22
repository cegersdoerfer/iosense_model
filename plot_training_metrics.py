import json
import sys
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_metrics(json_file_paths):
    # Initialize dictionaries to hold aggregated metrics
    aggregated_metrics = {
        'train_losses': defaultdict(list),
        'valid_losses': defaultdict(list),
        'train_f1': defaultdict(list),
        'valid_f1': defaultdict(list)
    }
    
    # Load and aggregate metrics from each JSON file
    for json_file_path in json_file_paths:
        if not os.path.isfile(json_file_path):
            print(f"Error: File '{json_file_path}' not found.")
            sys.exit(1)
        
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        for window_size in data.keys():
            model_data = data[window_size]
            aggregated_metrics['train_losses'][window_size].append(model_data.get("train_losses", []))
            aggregated_metrics['valid_losses'][window_size].append(model_data.get("valid_losses", []))
            aggregated_metrics['train_f1'][window_size].append(model_data.get("train_f1", []))
            aggregated_metrics['valid_f1'][window_size].append(model_data.get("valid_f1", []))
    
    # Function to calculate average metrics
    def calculate_average(metrics_list):
        return [sum(values) / len(values) for values in zip(*metrics_list)]
    
    # Calculate average metrics
    average_metrics = {
        'train_losses': {ws: calculate_average(losses) for ws, losses in aggregated_metrics['train_losses'].items()},
        'valid_losses': {ws: calculate_average(losses) for ws, losses in aggregated_metrics['valid_losses'].items()},
        'train_f1': {ws: calculate_average(f1_scores) for ws, f1_scores in aggregated_metrics['train_f1'].items()},
        'valid_f1': {ws: calculate_average(f1_scores) for ws, f1_scores in aggregated_metrics['valid_f1'].items()}
    }
    
    # Plotting function
    def plot_metric(metric_name, metric_data, ylabel):
        plt.figure(figsize=(10, 5))
        for window_size, values in metric_data.items():
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, label=f'Window {window_size}')
        plt.title(f'{metric_name} over Epochs for Different Window Sizes')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
    
    # Plot each metric
    plot_metric('Training Loss', average_metrics['train_losses'], 'Training Loss')
    plot_metric('Validation Loss', average_metrics['valid_losses'], 'Validation Loss')
    plot_metric('Training F1 Score', average_metrics['train_f1'], 'Training F1 Score')
    plot_metric('Validation F1 Score', average_metrics['valid_f1'], 'Validation F1 Score')

if __name__ == '__main__':
    json_file_paths = ['metrics.json']
    plot_metrics(json_file_paths)
