from read_csv import read_multiple_csv_folders
from matplotlib import pyplot as plt

train_results_folders = [   '/net/cremi/bbodis/espaces/travail/Rethinking_of_PAR/label_wise_metrics/train/RAP2/bald_0',
                            '/net/cremi/bbodis/espaces/travail/Rethinking_of_PAR/label_wise_metrics/train/RAP2/bald_1_only',
                            '/net/cremi/bbodis/espaces/travail/Rethinking_of_PAR/label_wise_metrics/train/RAP2/bald_2_only',
                            '/net/cremi/bbodis/espaces/travail/Rethinking_of_PAR/label_wise_metrics/train/RAP2/bald_3_only' ]

val_results_folders = [     '/net/cremi/bbodis/espaces/travail/Rethinking_of_PAR/label_wise_metrics/RAP2/bald_0',
                            '/net/cremi/bbodis/espaces/travail/Rethinking_of_PAR/label_wise_metrics/RAP2/bald_1_only',
                            '/net/cremi/bbodis/espaces/travail/Rethinking_of_PAR/label_wise_metrics/RAP2/bald_2_only',
                            '/net/cremi/bbodis/espaces/travail/Rethinking_of_PAR/label_wise_metrics/RAP2/bald_3_only' ]

baseline_val_results_folders = ['/net/cremi/bbodis/espaces/travail/Rethinking_of_PAR/label_wise_metrics/base_train_metric_stdout']
baseline_train_results_folders = ['/net/cremi/bbodis/espaces/travail/Rethinking_of_PAR/label_wise_metrics/base_train_metric_stdout/train']

attribute = "hs-BaldHead" # Specify the attribute here
scaling_amount = 100 # to scale down the std, as it is too big to be displayed when looking at specific attributes
metric = "ma"  # Specify the metric here


train_results = read_multiple_csv_folders(train_results_folders)
val_results = read_multiple_csv_folders(val_results_folders)
baseline_train_results = read_multiple_csv_folders(baseline_train_results_folders)
baseline_val_results = read_multiple_csv_folders(baseline_val_results_folders)

plt.figure()
plt.title(f"Mean of {metric} for each attribute - {attribute}")
plt.plot(train_results[metric].mean, 'b-o', label=f'train_{metric}', markersize=5)
plt.plot(val_results[metric].mean, 'r-o', label=f'val_{metric}', markersize=5)
plt.plot(train_results[metric].best_epoch, train_results[metric].mean[train_results[metric].best_epoch], 'bo', label=f'train_{metric}_best_epoch', markersize=10)
plt.plot(val_results[metric].best_epoch, val_results[metric].mean[val_results[metric].best_epoch], 'ro', label=f'val_{metric}_best_epoch', markersize=10)
plt.errorbar(range(len(train_results[metric].mean)), train_results[metric].mean, yerr=train_results[metric].std, fmt='b-o', markersize=5, label=f'train_{metric}_std', elinewidth=0.5, capsize=3)
plt.errorbar(range(len(val_results[metric].mean)), val_results[metric].mean, yerr=val_results[metric].std, fmt='r-o', markersize=5, label=f'val_{metric}_std', elinewidth=0.5, capsize=3)

# Print the standard deviation at each point
for i, std in enumerate(train_results[metric].std):
    plt.text(i, train_results[metric].mean[i] + std, f'{std:.2f}', ha='center', va='bottom', fontsize=8, color='blue')
for i, std in enumerate(val_results[metric].std):
    plt.text(i, val_results[metric].mean[i] + std, f'{std:.2f}', ha='center', va='bottom', fontsize=8, color='red')

# Plot baseline results without std in a darker color
plt.plot(baseline_train_results[metric].mean, 'b--', label=f'baseline_train_{metric}', markersize=5)
plt.plot(baseline_val_results[metric].mean, 'r--', label=f'baseline_val_{metric}', markersize=5)

plt.legend()
plt.ylim(0.6, 1)
plt.xlim(0, 9)
plt.grid(linewidth=1, linestyle='-', alpha=1)
plt.minorticks_on()  # Enable minor ticks
plt.grid(True, which='minor', linestyle='--', linewidth=0.5)  # Add minor gridlines
plt.show()



plt.figure()
plt.title(f"Mean and Std of {metric} for {attribute} attribute")
plt.plot(train_results[metric][attribute].mean, 'b-o', label=f'train_{metric}', markersize=5)
plt.plot(val_results[metric][attribute].mean, 'r-o', label=f'val_{metric}', markersize=5)
plt.plot(train_results[metric].best_epoch, train_results[metric][attribute].mean[train_results[metric].best_epoch], 'bo', label=f'train_{metric}_best_epoch', markersize=10)
plt.plot(val_results[metric].best_epoch, val_results[metric][attribute].mean[val_results[metric].best_epoch], 'ro', label=f'val_{metric}_best_epoch', markersize=10)
plt.errorbar(range(len(train_results[metric][attribute].mean)), train_results[metric][attribute].mean, yerr=train_results[metric][attribute].std/scaling_amount, fmt='b-o', markersize=5, label=f'train_{metric}_std', elinewidth=0.5, capsize=3)
plt.errorbar(range(len(val_results[metric][attribute].mean)), val_results[metric][attribute].mean, yerr=val_results[metric][attribute].std/scaling_amount, fmt='r-o', markersize=5, label=f'val_{metric}_std', elinewidth=0.5, capsize=3)

# Print the standard deviation at each point
for i, std in enumerate(train_results[metric][attribute].std):
    plt.text(i, train_results[metric][attribute].mean[i] + std/scaling_amount, f'{std:.2f}', ha='center', va='bottom', fontsize=8, color='blue')
for i, std in enumerate(val_results[metric][attribute].std):
    plt.text(i, val_results[metric][attribute].mean[i] + std/scaling_amount, f'{std:.2f}', ha='center', va='bottom', fontsize=8, color='red')

# Plot baseline results without std in a darker color
plt.plot(baseline_train_results[metric][attribute].mean, 'b--', label=f'baseline_train_{metric}', markersize=5)
plt.plot(baseline_val_results[metric][attribute].mean, 'r--', label=f'baseline_val_{metric}', markersize=5)

plt.legend()
plt.ylim(0.6, 1)
plt.xlim(0, 9)
plt.grid(linewidth=1, linestyle='-', alpha=1)
plt.minorticks_on()  # Enable minor ticks
plt.grid(True, which='minor', linestyle='--', linewidth=0.5)  # Add minor gridlines
plt.show()
