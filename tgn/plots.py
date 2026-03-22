import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define the models you want to plot and their corresponding result filenames
# Format: 'Label': 'filename_prefix'
models_to_plot = {
    'JODIE': 'jodie_rnn',
    'DyRep': 'dyrep_rnn',
    'TGN-attn': 'tgn-attn',
    'TGN-2l': 'tgn-2l',
    'TGN-mean': 'tgn-mean',
    'TGN-id': 'tgn-id',
    'TGN-no-mem': 'tgn-no-mem',
    'TGN-sum': 'tgn-sum',
    'TGN-time': 'tgn-time'
}
results_dir = Path('results')
plt.figure(figsize=(10, 6))

for label, prefix in models_to_plot.items():
    test_ap_list = []
    epoch_time_means = []
    
    # Logic from test.py:
    # 1. Load base file (e.g. tgn-id.pkl)
    # 2. Load numbered files (e.g. tgn-id_1.pkl to tgn-id_9.pkl)
    files_to_check = [f"{prefix}.pkl"] + [f"{prefix}_{i}.pkl" for i in range(1, 10)]
    
    for fname in files_to_check:
        file_path = results_dir / fname
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Extract metrics
                if 'test_ap' in data:
                    test_ap_list.append(data['test_ap'])
                
                # Handle time key (test.py uses 'epoch_times', previous plots.py used 'total_epoch_times')
                if 'epoch_times' in data:
                     epoch_time_means.append(np.mean(data['epoch_times']))
                elif 'total_epoch_times' in data:
                     epoch_time_means.append(np.mean(data['total_epoch_times']))
                     
            except Exception as e:
                print(f"Error loading {fname}: {e}")

    if not test_ap_list:
        print(f"Warning: No valid data found for {label} ({prefix})")
        continue

    # Calculate means and std devs
    avg_test_ap = np.mean(test_ap_list)
    std_test_ap = np.std(test_ap_list)
    
    avg_time = np.mean(epoch_time_means) if epoch_time_means else 0

    # Plotting
    y_val = avg_test_ap * 100
    y_err = std_test_ap * 100
    x_val = avg_time
    
    # Plot with error bars
    plt.errorbar(x_val, y_val, yerr=y_err, fmt='o', markersize=10, label=label, capsize=5)
    
    # Add text label near the point
    plt.annotate(f"{label}", (x_val, y_val), xytext=(0, -20), 
                 textcoords='offset points', ha='center')

    print(f"Plotted {label}: Time={avg_time:.2f}s, AP={y_val:.2f} +/- {y_err:.2f} (over {len(test_ap_list)} runs)")

plt.xlabel('Time (per epoch) in seconds', fontsize=12)
plt.ylabel('Test Average Precision', fontsize=12)
plt.title('TGN Variants Performance', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()