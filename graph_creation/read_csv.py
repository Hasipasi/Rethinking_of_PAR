
from matplotlib import pyplot as plt
import numpy as np
import os
import easydict as ED

#usage: python read_csv.py
 #read_csv_folder reads a single older containing csv files with the following format:
    #metric name in the filename
    #    first line: attribute index
    #    second line: attribute name
    #    third line-end: metric values for each attribute

    # results contain the following attributes:
    #    attrIdx: list of attribute indices
    #    attrName: list of attribute names
    #    metric (ma, f1, acc...): dict with the following attributes:
    #        all: list of all metric values for each attribute (2d list (epoch, attribute(by ID)))
    #        mean: mean metric value for each epoch (1d list (epoch))
    #        best_epoch: number of the epoch with the best mean metric value
    #        attrName: metric value for each epoch for the given attribute (1d list (epoch))

 #read_multiple_csv_folders reads multiple folders containing csv files with the following format:
    # metric name in the filename
    #    first line: attribute index
    #    second line: attribute name
    #    third line-end: metric values for each attribute

    # results contain the following attributes:
    #    attrIdx: list of attribute indices
    #    attrName: list of attribute names
    #    metric (ma, f1, acc...): dict with the following attributes:
    #        mean: mean metric value for each epoch (1d list (epoch)) averaged over all folders
    #        std: standard deviation of metric value for each epoch (1d list (epoch)) std over all folders
    #        best_epoch: number of the epoch with the best mean metric value averaged over all folders
    #        attrName: dict with the following attributes:
    #            mean: mean metric value for each epoch (1d list (epoch)) averaged over all folders
    #            std: standard deviation of metric value for each epoch (1d list (epoch)) std over all folders
    #   all: list of results for each folder in the same format as read_csv_folder

def read_csv_folder(results_folder, metrics=None):
    metrics = metrics or ['acc', 'prec', 'pos_recall','neg_recall', 'f1', 'ma']
    filenames = [f'label_{metric}.csv' for metric in metrics]
    filenames = [(results_folder +"\\"+ filename) for filename in filenames]
    results = ED.EasyDict()
    for i, metric in enumerate(metrics):
        filename = filenames[i]
        with open(filename, 'r') as f:
            lines = f.readlines()
        results.attrIdx = [int(idx) for idx in lines.pop(0).strip().split(';')[1:]]
        results.attrName = lines.pop(0).strip().split(';')[1:]
        results[metric] = ED.EasyDict()
        results[metric].all = [[float(result.replace(",",".")) for result in line[1:]] for line in [line.strip().split(';') for line in lines]]
        results[metric].mean = np.mean(results[metric].all, axis=1)
        results[metric].best_epoch = np.argmax(results[metric].mean)
        for attrIdx in results.attrIdx:
            attrName = results.attrName[attrIdx]
            results[metric][attrName] = np.array(results[metric].all)[:, attrIdx]
    return results

def read_multiple_csv_folders(results_folders, metrics=None):
    metrics = metrics or ['acc', 'prec', 'pos_recall','neg_recall', 'f1', 'ma']
    results = ED.EasyDict()
    results.all = []
    for i, results_folder in enumerate(results_folders):
        results.all.append(read_csv_folder(results_folder, metrics))

    for metric in metrics:
        results[metric] = ED.EasyDict()
        results[metric].mean = np.mean([results.all[i][metric].mean for i, _ in enumerate(results_folders)], axis=0)
        results[metric].std = np.std([results.all[i][metric].mean for i, _ in enumerate(results_folders)], axis=0)
        results[metric].best_epoch = np.argmax(results[metric].mean)
        for attridx in results.all[0].attrIdx:
            attrName = results.all[0].attrName[attridx]
            results[metric][attrName] = ED.EasyDict()
            results[metric][attrName].mean = np.mean([results.all[i][metric][attrName] for i, _ in enumerate(results_folders)], axis=0)
            results[metric][attrName].std = np.std([results.all[i][metric][attrName] for i, _ in enumerate(results_folders)], axis=0)
    results.attrIdx = results.all[0].attrIdx
    results.attrName = results.all[0].attrName
    return results

def main(): #example usage
    base_line_val = ['/net/cremi/bbodis/espaces/travail/Rethinking_of_PAR/label_wise_metrics']
    base_line_train = ['/net/cremi/bbodis/espaces/travail/Rethinking_of_PAR/label_wise_metrics/train']

    val_metrics = read_multiple_csv_folders(results_folders=base_line_val)
    train_metrics = read_multiple_csv_folders(results_folders=base_line_train)
    print("tooltip for attribute names:",val_metrics.attrName)
    plt.figure()
    for metric in ['ma']:
        
        plt.plot(val_metrics.all[0][metric].mean, 'r-o', label=f'val_{metric}', markersize=5)
        plt.plot(train_metrics.all[0][metric].mean, 'b-o', label=f'train_{metric}', markersize=5)
        plt.plot(val_metrics.all[0][metric].best_epoch, val_metrics[metric].mean[val_metrics[metric].best_epoch], 'ro', label=f'val_{metric}_best_epoch', markersize=10)
        # Plot results for ub-Jacket
        plt.plot(val_metrics.all[0][metric]["ub-Jacket"], 'm-o', label=f'val_{metric}_ub-Jacket', markersize=5)
        plt.plot(train_metrics.all[0][metric]["ub-Jacket"], 'c-o', label=f'train_{metric}_ub-Jacket', markersize=5)
        plt.legend()
    plt.title('Label wise metrics')
    plt.grid(True, which='both', linestyle='-', linewidth=1)
    plt.minorticks_on()  # Enable minor ticks
    plt.grid(True, which='minor', linestyle='--', linewidth=0.5)  # Add minor gridlines
    
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.ylim([0.7,1])
    plt.xlim([0,19])
    plt.show()


if __name__ == '__main__':
    main()