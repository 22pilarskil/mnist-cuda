import matplotlib.pyplot as plt
import re
import os
from pathlib import Path

def parse_log_file(filepath):
    epochs = []
    accuracies = []
    losses = []
    times = []
    
    with open(filepath, 'r') as file:
        for line in file:
            # Use regex to extract the numbers
            match = re.match(r'Epoch (\d+): Avg Accuracy = ([\d.]+), Avg Loss = ([\d.]+), Avg Time = ([\d.]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                accuracies.append(float(match.group(2)))
                losses.append(float(match.group(3)))
                times.append(float(match.group(4)))

    avg_time_per_epoch = sum(times) / len(times)
    
    return epochs, accuracies, losses, avg_time_per_epoch

def plot_training_curves(epochs, accuracies, losses, avg_time_per_epoch, output_path):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot accuracy on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(epochs, accuracies, color=color, marker='o', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Find and annotate the highest accuracy
    max_acc = max(accuracies)
    max_acc_epoch = epochs[accuracies.index(max_acc)]
    ax1.annotate(f'Max Acc: {max_acc:.3f}',
                 xy=(max_acc_epoch, max_acc),
                 xytext=(10, 10),
                 textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->'))
    
    # Create a second y-axis for loss
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color)
    ax2.plot(epochs, losses, color=color, marker='s', label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Find and annotate the lowest loss
    min_loss = min(losses)
    min_loss_epoch = epochs[losses.index(min_loss)]
    ax2.annotate(f'Min Loss: {min_loss:.3f}',
                 xy=(min_loss_epoch, min_loss),
                 xytext=(10, -20),
                 textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->'))
    
    # Add title and legend
    plt.title(f'Test Accuracy and Loss over Epochs (Rank {Path(output_path).stem}, Avg time/epoch: {avg_time_per_epoch:.2f}s)')
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
                epochs, accuracies, losses, avg_time_per_epoch = parse_log_file(filepath)
                plot_training_curves(epochs, accuracies, losses, avg_time_per_epoch, output_path)
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