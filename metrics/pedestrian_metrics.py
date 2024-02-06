import numpy as np
from easydict import EasyDict

remove_index = [38]
#reodered attribute indexes to match the order of the attributes in the dataset
#38 is removed as it contains the unused age attribute
#attributes with index 39 and above are adjusted accordingly
group_order = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
               36, 37, 38, 39, 40, 41, 42, 43, 44, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 45, 46, 47, 48, 49, 50, 51, 52, 53]
for remove_idx in remove_index:
    group_order.remove(remove_idx)
    group_order = [(i - 1 if i >= remove_idx else i) for i in group_order] 

def save_label_wise_metrics_to_csv(results, cfg=None, metrics = ['label_acc', 'label_f1', 'label_ma'], epoch=0, pkl_name='baseline', train=False, seed = None):
    """
    Save label-wise metrics to csv file
    """
    import csv
    import os
    root_path = "/home/bodis/Documents/GitHub/Rethinking_of_PAR/label_wise_metrics/"
    root_path += 'train/' if train else ''

    pkl_name += '_'+ str(seed)
    pkl_name = pkl_name.replace('.pkl', '')
    pkl_name = pkl_name.replace('data/', '')
    root_path = os.path.join(root_path, pkl_name)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    csv_paths = [os.path.join(root_path, csv_file+'.csv') for csv_file in metrics]
    print("Saving label-wise metrics to csv files:")   
    #append results to csv files
    for i, csv_path in enumerate(csv_paths): #for each metric
        metric = metrics[i]
        #before the first epoch, write the header
        if epoch == 0: 
            with open(csv_path, 'w', newline='') as f:
                header_row1 = ['Atrr_Idx']
                header_row2 = ['Attribute']
                for key, attr_name in enumerate(results.attr_names):
                    header_row1.append(key)
                    header_row2.append(attr_name)
                writer = csv.writer(f, delimiter=';')
                writer.writerow(header_row1)
                writer.writerow(header_row2)
            
        # append results to csv files
        with open(csv_path, 'a', newline='') as f:
            epoch_row = [epoch]
            for value in results[metric]:
                epoch_row.append(str(value).replace('.', ',')) #using comma as decimal separator
            writer = csv.writer(f, delimiter=';')
            writer.writerow(epoch_row)
        print(csv_path, "saved")


def get_pedestrian_metrics(gt_label, preds_probs, attr_names, threshold=0.5, index=None, cfg=None):
    """
    index: evaluated label index
    """
    pred_label = preds_probs > threshold

    eps = 1e-20
    result = EasyDict()

    if index is not None:
        pred_label = pred_label[:, index]
        gt_label = gt_label[:, index]

    ################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float) #shape (num_attributes,)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)  #shape (num_attributes,)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float) #shape (num_attributes,)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float) #shape (num_attributes,)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float) #shape (num_attributes,)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float) #shape (num_attributes,)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive #shape (num_attributes,)
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative #shape (num_attributes,)
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2 # mean accuracy #shape (num_attributes,)

    result.label_pos_recall = label_pos_recall #shape (num_attributes,)
    result.label_neg_recall = label_neg_recall #shape (num_attributes,)
    result.label_prec = true_pos / (true_pos + false_pos + eps) #shape (num_attributes,)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps) #shape (num_attributes,)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / ( 
            result.label_prec + result.label_pos_recall + eps) #shape (num_attributes,)

    result.label_ma = label_ma #shape (num_attributes,)
    result.ma = np.mean(label_ma) # mean accuracy #shape (1,)
    # get the attribute names in the order of the group_order and remove the batch dimension (duplicates)
    result.attr_names = [attr_names[idx][0] for idx in group_order]

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float) #shape (num_instances,)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float) # true positive #shape (num_instances,)

    intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float) #shape (num_instances,)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float) #shape (num_instances,)
    # accuracy
    instance_acc = intersect_pos / (union_pos + eps) #shape (num_instances,)
    result.instance_acc = np.mean(instance_acc) #shape (1,)
    # precision
    instance_prec = intersect_pos / (true_pos + eps) #shape (num_instances,)
    result.instance_prec = np.mean(instance_prec) #shape (1,)
    # recall
    instance_recall = intersect_pos / (gt_pos + eps) #shape (num_instances,)
    result.instance_recall = np.mean(instance_recall) #shape (1,)
    # F1
    result.instance_f1 = 2 * result.instance_prec * result.instance_recall / (result.instance_prec + result.instance_recall + eps) #shape (1,)

    # error, fn, fp
    result.error_num = false_pos + false_neg #shape (num_attributes,)
    result.fn_num = false_neg #shape (num_attributes,)
    result.fp_num = false_pos #shape (num_attributes,)
    return result 
