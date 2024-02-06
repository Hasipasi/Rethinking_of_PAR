from read_csv import read_multiple_csv_folders
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm
from format_graph import format_graph
import pandas as pd
import os


def plot_folder_with_mean_std(train_results, val_results, labels, title,
                              baseline_train_results=None, baseline_val_results=None,
                              attribute=None, metric='ma', scaling_amount=1,
                              ymin=0.8, alpha=0.8, mean_kerdojel=True):
    """
    Plot the folder with mean and standard deviation.
    """
    plt.figure()
    train_colors = cm.get_cmap('Blues')
    val_colors = cm.get_cmap('Reds')
    alpha = alpha if mean_kerdojel else 1
    
    if attribute:
        # Plot results for each attribute
        for i, results in enumerate(train_results.all):
            run_name = labels[i]
            plt.plot(results[metric][attribute], color=train_colors((i+1)/len(train_results.all)),  label=f'train_{metric}_{run_name}', markersize=5, alpha=alpha)
        for i, results in enumerate(train_results.all):
            run_name = labels[i]
            plt.plot(val_results.all[i][metric][attribute], color=val_colors((i+1)/len(train_results.all)), label=f'val_{metric}_{run_name}', markersize=5, alpha=alpha)
        
        if mean_kerdojel:
            # Plot averaged results with error bars
            plt.plot(train_results[metric][attribute].mean, 'b-o', label=f'train_{metric}_averaged_with_STD', markersize=7, linewidth=2)
            plt.plot(val_results[metric][attribute].mean, 'r-o', label=f'val_{metric}_averaged_with_STD', markersize=7, linewidth=2)
            plt.errorbar(range(len(train_results[metric][attribute].mean)), train_results[metric][attribute].mean, yerr=train_results[metric][attribute].std/scaling_amount, fmt='b-o', markersize=5,  elinewidth=0.5, capsize=3)
            plt.errorbar(range(len(val_results[metric][attribute].mean)), val_results[metric][attribute].mean, yerr=val_results[metric][attribute].std/scaling_amount, fmt='r-o', markersize=5, elinewidth=0.5, capsize=3)
            
            # Plot std values on top of the error bars
            for i, std in enumerate(train_results[metric][attribute].std):
                plt.text(i, train_results[metric][attribute].mean[i] + train_results[metric][attribute].std[i]/scaling_amount, f'{std:.3f}', ha='center', va='bottom', color='blue')
            for i, std in enumerate(val_results[metric][attribute].std):
                plt.text(i, val_results[metric][attribute].mean[i] + val_results[metric][attribute].std[i]/scaling_amount, f'{std:.3f}', ha='center', va='bottom', color='red')
        
        # Plot baseline results without std in a darker color
        if baseline_train_results:
            plt.plot(baseline_train_results[metric][attribute].mean, 'b--', label=f'averaged_baselines_train_{metric}')
        if baseline_val_results:
            plt.plot(baseline_val_results[metric][attribute].mean, 'r--', label=f'averaged_baselines_val_{metric}')
        

    else:
        # Plot results without attribute
        for i, results in enumerate(train_results.all):
            run_name = labels[i]
            plt.plot(results[metric].mean, color=train_colors((i+1)/len(train_results.all)),  label=f'train_{metric}_{run_name}', markersize=5, alpha=alpha)
        for i, results in enumerate(train_results.all):
            run_name = labels[i]
            plt.plot(val_results.all[i][metric].mean, color=val_colors((i+1)/len(train_results.all)), label=f'val_{metric}_{run_name}', markersize=5, alpha=alpha)
        
        if mean_kerdojel:
            # Plot averaged results with error bars
            plt.plot(train_results[metric].mean, 'b-o', label=f'train_{metric}_averaged_with_STD', markersize=7, linewidth=2)
            plt.plot(val_results[metric].mean, 'r-o', label=f'val_{metric}_averaged_with_STD', markersize=7, linewidth=2)
            plt.errorbar(range(len(train_results[metric].mean)), train_results[metric].mean, yerr=train_results[metric].std/scaling_amount, fmt='b-o', markersize=5,  elinewidth=0.5, capsize=3)
            plt.errorbar(range(len(val_results[metric].mean)), val_results[metric].mean, yerr=val_results[metric].std/scaling_amount, fmt='r-o', markersize=5, elinewidth=0.5, capsize=3)
            
            # Plot std values on top of the error bars
            for i, std in enumerate(train_results[metric].std):
                plt.text(i, train_results[metric].mean[i] + train_results[metric].std[i]/scaling_amount, f'{std:.3f}', ha='center', va='bottom', color='blue')
            for i, std in enumerate(val_results[metric].std):
                plt.text(i, val_results[metric].mean[i] + val_results[metric].std[i]/scaling_amount, f'{std:.3f}', ha='center', va='bottom', color='red')
        # Plot baseline results without std in a darker color         
        if baseline_train_results:
            plt.plot(baseline_train_results[metric].mean, 'b--', label=f'averaged_baselines_train_{metric}')
        if baseline_val_results:
            plt.plot(baseline_val_results[metric].mean, 'r--', label=f'averaged_baselines_val_{metric}')
    
    plt.legend(loc='upper left')  # Set the legend position to top left ????

    plt.title(title)
    num_epochs = len(train_results[metric].mean)
    format_graph(metric, num_epochs, ymin)
    plt.show()

def main():

    train_results_folders = [   'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new/train/RAP2/bald_0_605',
                                'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new/train/RAP2/bald_1_605',
                                'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new/train/RAP2/bald_2_605',
                                'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new/train/RAP2/bald_3_605',
                                'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new/train/baseline_605_1']


    val_results_folders = [     'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new/RAP2/bald_0_605',
                                'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new/RAP2/bald_1_605',
                                'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new/RAP2/bald_2_605',
                                'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new/RAP2/bald_3_605',
                                'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new/baseline_605_1']

        
    baseline_val_results_folders = ['C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\baseline_4_1',
                                    'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\baseline_69_1',
                                    'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\baseline_386_1',
                                    'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\baseline_523_1',
                                    'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\baseline_999_1',
                                    'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\baseline_1234_1',
                                    'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\baseline_605_1']
    
    baseline_train_results_folders = ['C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\train\\baseline_4_1',
                                      'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\train\\baseline_69_1',
                                      'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\train\\baseline_386_1',
                                      'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\train\\baseline_523_1',
                                      'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\train\\baseline_999_1',
                                      'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\train\\baseline_1234_1',
                                      'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\train\\baseline_605_1']

    #read results
    train_results = read_multiple_csv_folders(train_results_folders)
    val_results = read_multiple_csv_folders(val_results_folders)
    


    def get_run_name(folder, folders):
        if folder == folders[-1]:  # If this is the last folder
            return 'baseline_with_seed_605'
        else:
            s = folder.split('/')[-1].replace('baseline_', 'with_seed_').replace('_3', '_4000_synth').replace('_2', '_3000_synth').replace('_1', '_2000_synth').replace('_0', '_1000_synth').replace('_605', '')
            return 'ma_' + s
    #labels = [get_run_name(folder) for folder in train_results_folders]
    
    attribute = "ub-Jacket"  # Specify the attribute here hs-BaldHead   ub-Jacket lb-Skirt  ub-Shirt
    scaling_amount = 100 # to scale down the std, as it is too big to be displayed when looking at specific attributes
    metric = "ma"  # Specify the metric here
    mean_kerdojel = True # Plot the mean with error bars and std values on top of the error bars
    baseline_kerdojel = True # Plot the baseline 
    if baseline_kerdojel:
        baseline_train_results = read_multiple_csv_folders(baseline_train_results_folders)
        baseline_val_results = read_multiple_csv_folders(baseline_val_results_folders)
    else:    
        baseline_train_results = None
        baseline_val_results = None
    
    """plot_folder_with_mean_std(  train_results,
                                val_results,
                                labels,
                                title= 
                                f'MA metric for {attribute} attribute, using 4000 synthetic images compared to baseline results with mean and std across multiple generated datasets',
                                #"MA metric avaraged across all attributes for datasets including 4000 synthetic images, compared to baseline results",
                                #"MA metric avaraged across all attributes, using 4000 synthetic images compared to baseline results with mean and std across multiple generated datasets",
                                #f'MA metric for {attribute} attribute, using 1000, 2000, 3000, 4000 synthetic images compared to baseline results',
                                #"MA metric avaraged across all attributes, using 1000, 2000, 3000, 4000 synthetic images compared to baseline results",
                                baseline_train_results=baseline_train_results,
                                baseline_val_results=baseline_val_results,
                                attribute=attribute,
                                metric=metric,
                                scaling_amount=scaling_amount,
                                ymin=0.6,
                                alpha=0.3,
                                mean_kerdojel=mean_kerdojel)"""
    


    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    def plot_last_ma_scores(folders, title):
        fig, axs = plt.subplots(len(folders), 1, sharex=False, figsize=(10, len(folders)*5))
        
        for ax, folder in zip(axs, folders):
            # Load the data file. Adjust this line to fit your data format.
            data = pd.read_csv(os.path.join(folder, 'label_ma.csv'), sep=';')
            
            # Extract the last ma score for each attribute, excluding the first column (epoch number).
            last_ma_scores = data.iloc[-1, 1:]

            # Replace commas with periods and convert to float
            last_ma_scores = last_ma_scores.str.replace(',', '.').astype(float)

            # Plot the scores as a line with dots.
            ax.plot(last_ma_scores, marker='o', label=get_run_name(folder, folders))
            ax.set_ylim(0.7, 1)  # Set y-axis limits
            ax.legend(loc='lower right')

            # Add a grid to the background.
            ax.grid(True)

        plt.suptitle(title)
        plt.show()
    plot_last_ma_scores(train_results_folders, 'Ma for each attribute on the last epoch of training for increasing number of synthetic images')
    plot_last_ma_scores(val_results_folders, 'Ma for each attribute on the last epoch of validation for increasing number of synthetic images')

if __name__ == '__main__':
    main()