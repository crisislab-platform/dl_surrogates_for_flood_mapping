from modules.lib.constants import RUN_DIR

def plot_traning_metrics(history, run_dir):
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    # Plot training and validation loss
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)
    
    # Add reverse diffusion loss on secondary y-axis if available
    if 'sampling_loss' in history and history['sampling_loss']:
        sampling_loss = history['sampling_loss']
        ax2 = ax1.twinx()
        
        # Create x-axis positions for reverse_diff_loss (evenly spaced across epochs)
        num_epochs = len(history['loss'])
        num_rev_steps = len(sampling_loss)
        x_positions = np.linspace(0, num_epochs - 1, num_rev_steps)
        
        ax2.plot(x_positions, sampling_loss, label='Sampling Error', 
                color='tab:orange', linewidth=2, linestyle='--')
        ax2.set_ylabel('Sampling Error', fontsize=12, color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    else:
        ax1.legend(loc='upper right', fontsize=10)
    
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_validation_loss.png'), dpi=300)
    plt.close()
        
        