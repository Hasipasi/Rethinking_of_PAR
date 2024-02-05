import time

import numpy as np
from easydict import EasyDict
import torch

remove_index = [38]
group_order = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
               36, 37, 38, 39, 40, 41, 42, 43, 44, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 45, 46, 47, 48, 49, 50, 51, 52, 53]

for remove_idx in remove_index:
    group_order.remove(remove_idx)
    group_order = [(i - 1 if i >= remove_idx else i) for i in group_order] #readjust indexes past removed index


def get_pedestrian_metrics_labelwise(gt_label, preds_probs, attr_names, threshold=0.5, cfg=None):
    """
    Calculate metrics for each unique label in the ground truth labels
    """
    pred_label = preds_probs > threshold
    eps = 1e-20

    results = {}

    for index in range(gt_label.shape[1]):
        result = EasyDict()
        pred_label_wise = pred_label[index]
        gt_label_wise = gt_label[index]

        # label metrics
        gt_pos = np.sum((gt_label_wise == 1), axis=0).astype(float)
        gt_neg = np.sum((gt_label_wise == 0), axis=0).astype(float)
        true_pos = np.sum((gt_label_wise == 1) * (pred_label_wise == 1), axis=0).astype(float)
        true_neg = np.sum((gt_label_wise == 0) * (pred_label_wise == 0), axis=0).astype(float)
        false_pos = np.sum(((gt_label_wise == 0) * (pred_label_wise == 1)), axis=0).astype(float)
        false_neg = np.sum(((gt_label_wise == 1) * (pred_label_wise == 0)), axis=0).astype(float)

        label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
        label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
        label_ma = (label_pos_recall + label_neg_recall) / 2

        result.label_pos_recall = label_pos_recall
        result.label_neg_recall = label_neg_recall
        result.label_prec = true_pos / (true_pos + false_pos + eps)
        result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
        result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
                result.label_prec + result.label_pos_recall + eps)

        result.label_ma = label_ma
        result.ma = np.mean(label_ma)
        result.attr_name = attr_names[group_order[index]]

        results[index] = result

    return results

def save_label_wise_metrics_to_csv(results, cfg=None, csv_files = ['label_pos_recall', 'label_neg_recall', 'label_prec', 'label_acc', 'label_f1', 'label_ma'], epoch=0, pkl_name='baseline', train=False, seed = None):
    """
    Save label-wise metrics to csv file
    """
    import csv
    import os
    root_path = "/net/cremi/bbodis/espaces/travail/Rethinking_of_PAR/label_wise_metrics/"
    root_path += 'train/' if train else ''
    root_path
    pkl_name += '_'+ str(seed)
    pkl_name = pkl_name.replace('.pkl', '')
    pkl_name = pkl_name.replace('data/', '')
    root_path = os.path.join(root_path, pkl_name)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    csv_paths = [os.path.join(root_path, csv_file+'.csv') for csv_file in csv_files]
    print("Saving label-wise metrics to csv files:")   
    #append results to csv files
    for i, csv_path in enumerate(csv_paths):
        print(csv_path)
        if epoch == 0:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=';') #using semicolon as delimiter
                row = ['Atrr_Idx']
                for key, value in results.items():
                    row.append(key)
                writer.writerow(row)
                row = ['Attribute']
                for key, value in results.items():
                    row.append(value.attr_name[0])
                writer.writerow(row)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            metric = csv_files[i]
            row = [epoch]
            for key, value in results.items():
                row.append(str(value[metric]).replace('.', ',')) #using comma as decimal separator
            writer.writerow(row)



    # with open(csv_file, 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['label', 'label_pos_recall', 'label_neg_recall', 'label_prec', 'label_acc', 'label_f1', 'label_ma'])
    #     for key, value in results.items():
    #         writer.writerow([key, value.label_pos_recall, value.label_neg_recall, value.label_prec, value.label_acc, value.label_f1, value.label_ma])

def get_pedestrian_metrics(gt_label, preds_probs, threshold=0.5, index=None, cfg=None):
    """
    index: evaluated label index
    """
    pred_label = preds_probs > threshold

    eps = 1e-20
    result = EasyDict()

    if index is not None:
        pred_label = pred_label[:, index]
        gt_label = gt_label[:, index]

    ###############################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float)
    # true positive
    intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps)
    instance_prec = intersect_pos / (true_pos + eps)
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    # instance_f1 = np.mean(instance_f1)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    return result
