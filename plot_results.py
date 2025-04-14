import matplotlib.pyplot as plt
import re
import os
from pathlib import Path

def parse_log_file(filepath):
    epochs = []
    accuracies = []
    losses = []
    
    with open(filepath, 'r') as file:
        for line in file:
            # Use regex to extract the numbers
            match = re.match(r'Epoch (\d+): Avg Accuracy = ([\d.]+), Avg Loss = ([\d.]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                accuracies.append(float(match.group(2)))
                losses.append(float(match.group(3)))
    
    return epochs, accuracies, losses

def plot_training_curves(epochs, accuracies, losses, output_path):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot accuracy on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(epochs, accuracies, color=color, marker='o', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Create a second y-axis for loss
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color)
    ax2.plot(epochs, losses, color=color, marker='s', label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and legend
    plt.title(f'Training Accuracy and Loss over Epochs (Rank {Path(output_path).stem})')
    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
    
    # Adjust layout
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory
    print(f"Plot saved to {output_path}")

def process_directory(directory):
    # Create output directory if it doesn't exist
    output_dir = os.path.join(directory, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each .txt file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            output_path = os.path.join(output_dir, filename.replace('.txt', '.png'))
            
            try:
                epochs, accuracies, losses = parse_log_file(filepath)
                plot_training_curves(epochs, accuracies, losses, output_path)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python plot_training.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
    
    process_directory(directory)
    print("All plots generated successfully!")