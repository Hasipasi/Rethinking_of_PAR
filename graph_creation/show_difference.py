from read_csv import read_multiple_csv_folders
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm
from format_graph import format_graph

def plot_folder_difference(train_results, val_results, labels, title,
                              baseline_train_results, baseline_val_results,
                              attribute=None, metric='ma'):
    """
    Plot the folder with mean and standard deviation.
    """
    plt.figure()
    train_colors = cm.get_cmap('Blues')
    val_colors = cm.get_cmap('Reds')
    num_epochs = len(train_results[metric].mean)
    plt.plot([0, num_epochs], [0,0], '-', linewidth=2, color='gray')
    
    if attribute:
        # Plot results for each attribute
        base_train = np.array(baseline_train_results[metric][attribute].mean)
        base_val = np.array(baseline_val_results[metric][attribute].mean)
        
        for i, results in enumerate(train_results.all):
            run_name = labels[i]
            results = np.array(results[metric][attribute])
            results = results - base_train
            plt.plot(results, color=train_colors((i+1)/len(train_results.all)),  label=f'train_{metric}_{run_name}', markersize=5, alpha=1)
        for i, results in enumerate(val_results.all):
            run_name = labels[i]
            results = np.array(results[metric][attribute])
            results = results - base_val
            plt.plot(results, color=val_colors((i+1)/len(train_results.all)), label=f'val_{metric}_{run_name}', markersize=5, alpha=1)
        
        
    else:
        # Plot results without attribute
        base_train = baseline_train_results[metric].mean
        base_val = baseline_val_results[metric].mean
        
        for i, results in enumerate(train_results.all):
            run_name = labels[i]
            plt.plot(results[metric].mean, color=train_colors((i+1)/len(train_results.all)),  label=f'train_{metric}_{run_name}', markersize=5, alpha=1)
        for i, results in enumerate(train_results.all):
            run_name = labels[i]
            plt.plot(val_results.all[i][metric].mean, color=val_colors((i+1)/len(train_results.all)), label=f'val_{metric}_{run_name}', markersize=5, alpha=1)
        
    # Plot baseline results without std in a darker color         
    
    

    plt.title(title)

    format_graph(metric, num_epochs, -0.3, 0.3)
    plt.show()

def main():
    # train_results_folders = [   '/Users/bodisbalazs/Desktop/visualization/label_wise_metrics/baseline_4',
    #                             '/Users/bodisbalazs/Desktop/visualization/label_wise_metrics/baseline_69',
    #                             '/Users/bodisbalazs/Desktop/visualization/label_wise_metrics/baseline_386',
    #                             '/Users/bodisbalazs/Desktop/visualization/label_wise_metrics/baseline_523',
    #                             '/Users/bodisbalazs/Desktop/visualization/label_wise_metrics/baseline_999',
    #                             '/Users/bodisbalazs/Desktop/visualization/label_wise_metrics/baseline_1234',
    #                             '/Users/bodisbalazs/Desktop/visualization/label_wise_metrics/baseline_605']

    # val_results_folders = [     '/Users/bodisbalazs/Desktop/visualization/label_wise_metrics/train/baseline_4',
    #                             '/Users/bodisbalazs/Desktop/visualization/label_wise_metrics/train/baseline_69',
    #                             '/Users/bodisbalazs/Desktop/visualization/label_wise_metrics/train/baseline_386',
    #                             '/Users/bodisbalazs/Desktop/visualization/label_wise_metrics/train/baseline_523',
    #                             '/Users/bodisbalazs/Desktop/visualization/label_wise_metrics/train/baseline_999',
    #                             '/Users/bodisbalazs/Desktop/visualization/label_wise_metrics/train/baseline_1234',
    #                             '/Users/bodisbalazs/Desktop/visualization/label_wise_metrics/train/baseline_605']
    
    val_results_folders = ['C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\baseline_4_1',
                                    'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\baseline_69_1',
                                    'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\baseline_386_1',
                                    'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\baseline_523_1',
                                    'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\baseline_999_1',
                                    'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\baseline_1234_1']

    train_results_folders = [ 'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\train\\baseline_4_1',
                                      'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\train\\baseline_69_1',
                                      'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\train\\baseline_386_1',
                                      'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\train\\baseline_523_1',
                                      'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\train\\baseline_999_1',
                                      'C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\train\\baseline_1234_1']
    
    baseline_val_results_folders = ['C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\baseline_605_1']
    baseline_train_results_folders = ['C:\\Users\\budai\\Downloads\\visualization\\visualization\\label_wise_metrics_new\\train\\baseline_605_1']

    #read results
    train_results = read_multiple_csv_folders(train_results_folders)
    val_results = read_multiple_csv_folders(val_results_folders)
    

    def get_run_name(folder):
        return folder.split('/')[-1].replace('baseline_', 'with_seed_')
    
    labels = [get_run_name(folder) for folder in train_results_folders]
    
    attribute =  'hs-BaldHead' # Specify the attribute here
    scaling_amount = 1 # to scale down the std, as it is too big to be displayed when looking at specific attributes
    metric = "ma"  # Specify the metric here
    mean_kerdojel = False # Plot the mean with error bars and std values on top of the error bars
    baseline_kerdojel = True # Plot the baseline 
    if baseline_kerdojel:
        baseline_train_results = read_multiple_csv_folders(baseline_train_results_folders)
        baseline_val_results = read_multiple_csv_folders(baseline_val_results_folders)
    else:    
        baseline_train_results = None
        baseline_val_results = None
    
    plot_folder_difference(  train_results,
                                val_results,
                                labels,
                                title="MA metric difference of each run from the baseline",
                                baseline_train_results=baseline_train_results,
                                baseline_val_results=baseline_val_results,
                                attribute=attribute,
                                metric=metric)

if __name__ == '__main__':
    main() 