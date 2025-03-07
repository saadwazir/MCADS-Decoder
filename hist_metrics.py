
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import tensorflow as tf
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage
from cupyx.scipy.spatial.distance import cdist
# from scipy.spatial.distance import cdist
import random
import pandas as pd
import shutil
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
import subprocess
import time
import datetime
import seg_metrics.seg_metrics as sg





from medpy.metric.binary import dc as mpdc
from medpy.metric.binary import jc as mpjc
from medpy.metric.binary import hd as mphd
from medpy.metric.binary import asd as mpasd
from medpy.metric.binary import specificity as mpspecificity
from medpy.metric.binary import sensitivity as mpsensitivity
from medpy.metric.binary import precision as mpprecision
from medpy.metric.binary import recall as mprecall

seed = 13334
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)
cp.random.seed(seed=seed)


def conf_matrix_keras_iou_manual(y_test, pred):

    m = tf.keras.metrics.BinaryIoU()
    m.reset_state()
    m.update_state(y_test, pred)
    print("----------------------------------")
    values = np.array(m.get_weights()).reshape(2,2)
    print(values)
    print("Keras mIoU: ", m.result().numpy() )

    #IOU for each class is..
    # IOU = true_positive / (true_positive + false_positive + false_negative).

    conf_matrix = values

    # To calculate IoU for each class
    values = conf_matrix

    class0_iou = values[0, 0] / (values[0, 0] + values[0, 1] + values[1, 0])
    class1_iou = values[1, 1] / (values[1, 1] + values[1, 0] + values[0, 1])

    print("IoU for Class 0:", class0_iou)
    print("IoU for Class 1:", class1_iou)


def miou_keras_class_0(y_test, pred, threshold):
    m = tf.keras.metrics.BinaryIoU(target_class_ids=[0], threshold=threshold)
    m.reset_state()
    m.update_state(y_test, pred)
    return m.result().numpy()

def miou_keras_class_1(y_test, pred, threshold):
    m = tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=threshold)
    m.reset_state()
    m.update_state(y_test, pred)
    return m.result().numpy()




def keras_bin_miou_per_image(y_test, pred):
    
    def iou_keras_def(mask1, mask2):
    
        m = tf.keras.metrics.BinaryIoU(target_class_ids=[1])
        m.reset_state()
        m.update_state(mask1, mask2)
        iou = m.result().numpy()
        return iou
    
    iou_score_each_sample = []
    for i in range(0, y_test.shape[0]):
        d = iou_keras_def(y_test[i], pred[i])
        iou_score_each_sample.append(d)

    return np.mean(iou_score_each_sample)




# def np_miou_per_image(y_test, pred):
    
#     def calculateIoU(gtMask, predMask):
        
#         predMask = (predMask > 0.5)
#         predMask = np.where(predMask>0, 1, 0)
        
#         # Calculate the true positives,
#         # false positives, and false negatives
#         tp = 0
#         fp = 0
#         fn = 0

#         for i in range(len(gtMask)):
#             for j in range(len(gtMask[0])):
#                 if gtMask[i][j] == 1 and predMask[i][j] == 1:
#                     tp += 1
#                 elif gtMask[i][j] == 0 and predMask[i][j] == 1:
#                     fp += 1
#                 elif gtMask[i][j] == 1 and predMask[i][j] == 0:
#                     fn += 1

#         # Calculate IoU
#         iou = tp / (tp + fp + fn)

#         return iou


#     iou_score_each_sample = []

#     for i in range(0, y_test.shape[0]):
#         d = calculateIoU(y_test[i], pred[i])
#         iou_score_each_sample.append(d)

#     return np.mean(iou_score_each_sample)

def calculateIoU(gtMask, predMask):
    """Calculate IoU for a single image using Cupy."""
    # predMask = cp.where(predMask > threshold, 1, 0)
    
    tp = cp.sum((gtMask == 1) & (predMask == 1))
    fp = cp.sum((gtMask == 0) & (predMask == 1))
    fn = cp.sum((gtMask == 1) & (predMask == 0))
    iou = tp / (tp + fp + fn)
    return iou

def np_miou_per_image(y_test, pred):
    
    y_test = cp.asarray(y_test)
    pred = cp.asarray(pred)
    
    """Compute mean IoU per image over a batch using Cupy."""
    iou_scores = cp.zeros(y_test.shape[0], dtype=cp.float32)
    for i in range(y_test.shape[0]):
        iou_scores[i] = calculateIoU(y_test[i], pred[i])
    
    result = cp.mean(iou_scores).get()  # Convert to NumPy for compatibility if needed
    
    result = result.astype(np.float32).item()
    
    return round(result,4) 





# def np_dice_per_image(y_test, pred):
    
#     pred = (pred > 0.5)
#     pred = np.where(pred>0, 1, 0)

#     def DICE_COE(mask1, mask2):
#         intersect = np.sum(mask1*mask2)
#         fsum = np.sum(mask1)
#         ssum = np.sum(mask2)
#         dice = (2 * intersect ) / (fsum + ssum)
#         #dice = np.mean(dice)
#         #dice = round(dice, 2) # for easy reading
#         return dice



#     dice_score_each_sample = []

#     for i in range(0, y_test.shape[0]):
#         d = DICE_COE(y_test[i], pred[i])
#         dice_score_each_sample.append(d)

#     return np.mean(dice_score_each_sample)



def DICE_COE(mask1, mask2):
    """Calculate Dice coefficient for a single image using Cupy."""
    intersect = cp.sum(mask1 * mask2)
    fsum = cp.sum(mask1)
    ssum = cp.sum(mask2)
    dice = (2. * intersect) / (fsum + ssum)
    return dice

def np_dice_per_image(y_test, pred):

    y_test = cp.asarray(y_test)
    pred = cp.asarray(pred)
    
    """Compute mean Dice coefficient per image over a batch using Cupy."""
    dice_scores = cp.zeros(y_test.shape[0], dtype=cp.float32)
    for i in range(y_test.shape[0]):
        dice_scores[i] = DICE_COE(y_test[i], pred[i])
    
    result = cp.mean(dice_scores).get()
    
    result = result.astype(np.float32).item()

    return round(result,4)




def calculateHD(gtMask, predMask):
    """
    Calculate the Hausdorff Distance (HD) for a single image using CuPy.
    """
    gtBoundary = cp.argwhere(gtMask)
    predBoundary = cp.argwhere(predMask)
    
    if len(gtBoundary) == 0 or len(predBoundary) == 0:
        return cp.inf
    
    distances = cdist(gtBoundary, predBoundary, metric='euclidean')
    
    max_dist_gt_to_pred = cp.max(cp.min(distances, axis=1))
    max_dist_pred_to_gt = cp.max(cp.min(distances, axis=0))
    
    hd = cp.max(cp.array([max_dist_gt_to_pred, max_dist_pred_to_gt]))
    
    return hd.get()

def np_hd_per_image(y_test, pred):
    """
    Compute Hausdorff Distance (HD) per image over a batch using CuPy.
    """
    # Ensure the input arrays are CuPy arrays
    y_test_cp = cp.asarray(y_test)
    pred_cp = cp.asarray(pred)
    
    hd_scores = cp.zeros(y_test_cp.shape[0], dtype=cp.float64)
    for i in range(y_test_cp.shape[0]):
        hd_scores[i] = calculateHD(y_test_cp[i], pred_cp[i])
    
    valid_scores = hd_scores[cp.isfinite(hd_scores)]
    
    if len(valid_scores) == 0:
        return float('inf')
    
    mean_hd = cp.mean(valid_scores).get()  # Convert to Python float for compatibility
    # return mean_hd[0]
    
    return round(mean_hd.item(), 4)







def sm_iou_score_per_image(y_test, pred, threshold):
    
    sm_iou = sm.metrics.IOUScore(per_image = True, threshold = threshold)
    return sm_iou(y_test, pred).numpy()


def sm_iou_score_whole_batch(y_test, pred, threshold):
    sm_iou = sm.metrics.IOUScore(per_image = False, threshold = threshold)
    return sm_iou(y_test, pred).numpy()


def sm_f1_score_per_image(y_test, pred, threshold):
    sm_f1 = sm.metrics.FScore(per_image = True, threshold = threshold)
    return sm_f1(y_test, pred).numpy()


def sm_f1_score_whole_batch(y_test, pred, threshold):
    sm_f1 = sm.metrics.FScore(per_image = False, threshold = threshold)
    return sm_f1(y_test, pred).numpy()


def sm_precision_whole_batch(y_test, pred, threshold):
    sm_precision = sm.metrics.Precision(per_image = False, threshold = threshold)
    return sm_precision(y_test, pred).numpy()


def sm_precision_per_image(y_test, pred, threshold):
    sm_precision = sm.metrics.Precision(per_image = True, threshold = threshold)
    return sm_precision(y_test, pred).numpy()


def sm_recall_whole_batch(y_test, pred, threshold):
    sm_recall = sm.metrics.Recall(per_image = False, threshold = threshold)
    return sm_recall(y_test, pred).numpy()


def sm_recall_per_image(y_test, pred, threshold):
    sm_recall = sm.metrics.Recall(per_image = True, threshold = threshold)
    return sm_recall(y_test, pred).numpy()




def medpy_dice(y_test, pred):
    
    results_mpdc = mpdc(y_test, pred)
    return results_mpdc


def medpy_jc(y_test, pred):
    
    results_mpjc = mpjc(y_test, pred)
    return results_mpjc


def medpy_hd(y_test, pred):
    
    results_mphd = mphd(y_test, pred)
    return results_mphd


def medpy_asd(y_test, pred):
    
    results_mpasd = mpasd(y_test, pred)
    return results_mpasd


def medpy_specificity(y_test, pred):
    
    results_mpspecificity = mpspecificity(y_test, pred)
    return results_mpspecificity


def medpy_sensitivity(y_test, pred):
    
    results_mpsensitivity = mpsensitivity(y_test, pred)
    return results_mpsensitivity


def medpy_precision(y_test, pred):
        
    results_mpprecision= mpprecision(y_test, pred)
    return results_mpprecision


def medpy_recall(y_test, pred):
    
    results_mprecall= mprecall(y_test, pred)
    return results_mprecall


def np_p_r_f1(y_test, pred):
    def calculate_metrics_per_image(y_true, y_pred):
        num_images = y_true.shape[0]
        total_precision, total_recall, total_f1_score = 0, 0, 0

        for i in range(num_images):
            # Flatten the arrays for the i-th image
            y_true_flatten = y_true[i].flatten()
            y_pred_flatten = y_pred[i].flatten()

            # True Positives, False Positives, False Negatives, True Negatives
            tp = np.sum((y_true_flatten == 1) & (y_pred_flatten == 1))
            fp = np.sum((y_true_flatten == 0) & (y_pred_flatten == 1))
            fn = np.sum((y_true_flatten == 1) & (y_pred_flatten == 0))

            # Precision, Recall, F1 Score for the i-th image
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            total_precision += precision
            total_recall += recall
            total_f1_score += f1_score

        # Calculate average metrics
        avg_precision = total_precision / num_images
        avg_recall = total_recall / num_images
        avg_f1_score = total_f1_score / num_images

        return avg_precision, avg_recall, avg_f1_score

    return calculate_metrics_per_image(y_test, pred)




def np_specificity_np_sensitivity(y_test, pred):

    def calculate_average_sensitivity_specificity(y_test, pred):
        
        total_sensitivity = 0
        total_specificity = 0
        num_images = y_test.shape[0]

        for i in range(num_images):
            y_test_flat = y_test[i].flatten()
            pred_flat = pred[i].flatten()

            # Calculating True Positives, False Positives, True Negatives, and False Negatives
            TP = np.sum((y_test_flat == 1) & (pred_flat == 1))
            FP = np.sum((y_test_flat == 0) & (pred_flat == 1))
            TN = np.sum((y_test_flat == 0) & (pred_flat == 0))
            FN = np.sum((y_test_flat == 1) & (pred_flat == 0))

            # Calculating Sensitivity and Specificity for each image
            sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

            total_sensitivity += sensitivity
            total_specificity += specificity

        # Calculating average sensitivity and specificity
        average_sensitivity = total_sensitivity / num_images
        average_specificity = total_specificity / num_images

        return average_sensitivity, average_specificity

    # Example usage
    average_sensitivity, average_specificity = calculate_average_sensitivity_specificity(y_test, pred)
    # print("Average Sensitivity:", average_sensitivity)
    # print("Average Specificity:", average_specificity)
    
    return [average_specificity, average_sensitivity]


def np_object_p_r_f2(y_test, pred, iou_threshold):

    def label_objects_cupy(segmentation_map):
        """
        Label individual objects in a binary segmentation map using Cupyx.
        """
        labeled_map, num_features = cupyx.scipy.ndimage.label(segmentation_map)
        return labeled_map, num_features

    def calculate_iou_cupy(predicted_map, ground_truth_map, predicted_label, ground_truth_label):
        """
        Calculate the Intersection over Union (IoU) for a predicted object and a ground truth object.
        """
        predicted_object = (predicted_map == predicted_label)
        ground_truth_object = (ground_truth_map == ground_truth_label)
        
        intersection = cp.logical_and(predicted_object, ground_truth_object).sum()
        union = cp.logical_or(predicted_object, ground_truth_object).sum()
        
        return intersection / union if union != 0 else 0

    def match_objects(predicted_map, ground_truth_map, iou_threshold):
        """
        Match predicted objects to ground truth objects based on IoU threshold.
        """
        predicted_labels = cp.unique(predicted_map)[1:]  # Exclude background
        ground_truth_labels = cp.unique(ground_truth_map)[1:]  # Exclude background
        
        matches = []
        unmatched_predicted = {int(label) for label in predicted_labels.tolist()}  # Convert to set of integers
        unmatched_ground_truth = {int(label) for label in ground_truth_labels.tolist()}  # Convert to set of integers
        
        for p_label in predicted_labels:
            best_match = None
            best_iou = iou_threshold
            for gt_label in ground_truth_labels:
                iou = calculate_iou_cupy(predicted_map, ground_truth_map, int(p_label), int(gt_label))
                if iou > best_iou:
                    best_match = (int(p_label), int(gt_label))
                    best_iou = iou
            if best_match:
                matches.append(best_match)
                unmatched_predicted.discard(best_match[0])
                unmatched_ground_truth.discard(best_match[1])
        
        return matches, unmatched_predicted, unmatched_ground_truth

    def calculate_metrics_for_batch(y_test, pred, iou_threshold):
        """
        Calculate object-level precision, recall, and F1 score for a batch of images.
        """
        batch_size = pred.shape[0]
        precision_list = []
        recall_list = []
        f1_score_list = []
        
        for i in range(batch_size):
            predicted_map = cp.asarray(pred[i, ..., 0])  # Convert to Cupy array and remove channel dimension
            ground_truth_map = cp.asarray(y_test[i, ..., 0])  # Convert to Cupy array and remove channel dimension
            
            predicted_labeled_map, _ = label_objects_cupy(predicted_map)
            ground_truth_labeled_map, _ = label_objects_cupy(ground_truth_map)
            
            matches, unmatched_predicted, unmatched_ground_truth = match_objects(predicted_labeled_map, ground_truth_labeled_map, iou_threshold)
            
            tp = len(matches)
            fp = len(unmatched_predicted)
            fn = len(unmatched_ground_truth)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)
        
        # Calculate average metrics across the batch
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1_score = np.mean(f1_score_list)
        
        return avg_precision, avg_recall, avg_f1_score


    avg_precision, avg_recall, avg_f1_score = calculate_metrics_for_batch(y_test, pred, iou_threshold)
    return [avg_precision, avg_recall, avg_f1_score]



def seg_metrics_fuc(y_test, pred):
    
    dice_arr = []
    jaccard_arr = []
    precision_arr = []
    recall_arr = []
    fpr_arr = []
    fnr_arr = []
    vs_arr = []
    hd_arr = []
    hd95_arr = []
    msd_arr = []
    mdsd_arr = []
    stdsd_arr = []
    tp_arr = []
    tn_arr = []
    fp_arr = []
    fn_arr = []
    false_omission_rate_arr = []
    accuracy_arr = []
    
    
    
    
    for i in range(y_test.shape[0]):
        
        gdth_img = y_test[i]
        pred_img = pred[i]


        labels = [1]
        csv_file = 'metrics.csv'

        metrics = sg.write_metrics(labels=labels,  # exclude background if needed
                        gdth_img=gdth_img,
                        pred_img=pred_img,
                        TPTNFPFN=True,
                        metrics=['dice','jaccard','precision','recall','fpr','fnr','vs','hd','hd95','msd','mdsd','stdsd','TP','TN','FP','FN'])


        metrics = metrics[0]
        metrics = {k: float(v[0]) for k, v in metrics.items()}
        # print(metrics)
        dice, jaccard, precision, recall ,fpr ,fnr ,vs ,hd ,hd95 ,msd ,mdsd ,stdsd , tp, tn, fp, fn = metrics['dice'], metrics['jaccard'], metrics['precision'], metrics['recall'], metrics['fpr'], metrics['fnr'], metrics['vs'], metrics['hd'], metrics['hd95'], metrics['msd'], metrics['mdsd'], metrics['stdsd'], metrics['TP'] , metrics['TN'] , metrics['FP'] , metrics['FN']
        false_omission_rate = fn/(fn+tn)
        accuracy = (tp + tn)/(tp + tn + fp + fn)
        
        # print(false_omission_rate)
        # print(accuracy)
        
        dice_arr.append(dice)
        jaccard_arr.append(jaccard)
        precision_arr.append(precision)
        recall_arr.append(recall)
        fpr_arr.append(fpr)
        fnr_arr.append(fnr)
        vs_arr.append(vs)
        hd_arr.append(hd)
        hd95_arr.append(hd95)
        msd_arr.append(msd)
        mdsd_arr.append(mdsd)
        stdsd_arr.append(stdsd)
        tp_arr.append(tp)
        tn_arr.append(tn)
        fp_arr.append(fp)
        fn_arr.append(fn)
        false_omission_rate_arr.append(false_omission_rate)
        accuracy_arr.append(accuracy)
    
    total_n = y_test.shape[0]
    

    dice_arr = np.array(dice_arr).mean()
    jaccard_arr = np.array(jaccard_arr).mean()
    precision_arr = np.array(precision_arr).mean()
    recall_arr = np.array(recall_arr).mean()
    fpr_arr = np.array(fpr_arr).mean()
    fnr_arr = np.array(fnr_arr).mean()
    vs_arr = np.array(vs_arr).mean()
    hd_arr = np.array(hd_arr).mean()
    hd95_arr = np.array(hd95_arr).mean()
    msd_arr = np.array(msd_arr).mean()
    mdsd_arr = np.array(mdsd_arr).mean()
    stdsd_arr = np.array(stdsd_arr).mean()
    tp_arr = np.array(tp_arr).mean()
    tn_arr = np.array(tn_arr).mean()
    fp_arr = np.array(fp_arr).mean()
    fn_arr = np.array(fn_arr).mean()
    false_omission_rate_arr = np.array(false_omission_rate_arr).mean()
    accuracy_arr = np.array(accuracy_arr).mean()
    
    return dice_arr, jaccard_arr, precision_arr, recall_arr, fpr_arr, fnr_arr, vs_arr, hd_arr, hd95_arr, msd_arr, mdsd_arr, stdsd_arr, tp_arr, tn_arr, fp_arr, fn_arr, false_omission_rate_arr, accuracy_arr




def eval_all_metrics(y_test, pred, all = 0, threshold_pred = 0.5):
    
    pred = np.where(pred>threshold_pred, 1, 0)
    pred = pred.astype(np.float32)
    
    df = pd.DataFrame(columns=['Metric', 'Score'])

    try:
        record = {'Metric': "miou_keras_class_0", 'Score': miou_keras_class_0(y_test, pred, threshold_pred)}
    except:
        record = {'Metric': "miou_keras_class_0", 'Score': -1}
    df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)


    try:
        record = {'Metric': "miou_keras_class_1", 'Score': miou_keras_class_1(y_test, pred, threshold_pred)}
    except:
        record = {'Metric': "miou_keras_class_1", 'Score': -1}
    df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    
    try:
        record = {'Metric': "sm_iou_score_whole_batch", 'Score': sm_iou_score_whole_batch(y_test, pred, threshold_pred)}
    except:
        record = {'Metric': "sm_iou_score_whole_batch", 'Score': -1}
    df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)

    if all == 1:
        
        try:
            record = {'Metric': "medpy_jc", 'Score': medpy_jc(y_test, pred)}
        except:
            record = {'Metric': "medpy_jc", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)

    
    try:
        record = {'Metric': "keras_bin_miou_per_image", 'Score': keras_bin_miou_per_image(y_test, pred)}
    except:
        record = {'Metric': "keras_bin_miou_per_image", 'Score': -1}
    df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)

    
    try:
        record = {'Metric': "sm_iou_score_per_image", 'Score': sm_iou_score_per_image(y_test, pred, threshold_pred)}
    except:
        record = {'Metric': "sm_iou_score_per_image", 'Score': -1}
    df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)

    if all == 1:
        try:
            record = {'Metric': "np_miou_per_image", 'Score': np_miou_per_image(y_test, pred)}
        except:
            record = {'Metric': "np_miou_per_image", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    if all == 1:
        
        try:
            record = {'Metric': "np_dice_per_image", 'Score': np_dice_per_image(y_test, pred)}
        except:
            record = {'Metric': "np_dice_per_image", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)

    if all == 1:
        
        try:
            record = {'Metric': "sm_f1_score_per_image", 'Score': sm_f1_score_per_image(y_test, pred, threshold_pred)}
        except:
            record = {'Metric': "sm_f1_score_per_image", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    if all == 1:
        
        try:
            record = {'Metric': "np_f1_score_per_image", 'Score': np_p_r_f1(y_test, pred)[2]}
        except:
            record = {'Metric': "np_f1_score_per_image", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)

    if all == 1:
        
        try:
            record = {'Metric': "sm_f1_score_whole_batch", 'Score': sm_f1_score_whole_batch(y_test, pred, threshold_pred)}
        except:
            record = {'Metric': "sm_f1_score_whole_batch", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)

    if all == 1:
        
        try:
            record = {'Metric': "medpy_dice", 'Score': medpy_dice(y_test, pred)}
        except:
            record = {'Metric': "medpy_dice", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)

    if all == 1:
        
        try:
            record = {'Metric': "medpy_hd", 'Score': medpy_hd(y_test, pred)}
        except:
            record = {'Metric': "medpy_hd", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        

    # if all == 1:
    #     try:
    #         record = {'Metric': "np_hd_per_image", 'Score': np_hd_per_image(y_test, pred)}
    #     except:
    #         record = {'Metric': "np_hd_per_image", 'Score': -1}
    #     df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    
    if all == 1:
        
        try:
            record = {'Metric': "medpy_asd", 'Score': medpy_asd(y_test, pred)}
        except:
            record = {'Metric': "medpy_asd", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    # record = {'Metric': "medpy_specificity", 'Score': medpy_specificity(y_test, pred)}
    # df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    # record = {'Metric': "medpy_sensitivity", 'Score': medpy_sensitivity(y_test, pred)}
    # df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    # record = {'Metric': "medpy_precision", 'Score': medpy_precision(y_test, pred)}
    # df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    # record = {'Metric': "medpy_recall", 'Score': medpy_recall(y_test, pred)}
    # df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    if all == 1:
        
        try:
            record = {'Metric': "sm_precision_whole_batch", 'Score': sm_precision_whole_batch(y_test, pred, threshold_pred)}
        except:
            record = {'Metric': "sm_precision_whole_batch", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    if all == 1:
        
        try:
            record = {'Metric': "sm_precision_per_image", 'Score': sm_precision_per_image(y_test, pred, threshold_pred)}
        except:
            record = {'Metric': "sm_precision_per_image", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    if all == 1:
        
        try:
            record = {'Metric': "np_precision_per_image", 'Score': np_p_r_f1(y_test, pred)[0]}
        except:
            record = {'Metric': "np_precision_per_image", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    if all == 1:
        
        try:
            record = {'Metric': "sm_recall_whole_batch", 'Score': sm_recall_whole_batch(y_test, pred, threshold_pred)}
        except:
            record = {'Metric': "sm_recall_whole_batch", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    if all == 1:
        
        try:
            record = {'Metric': "sm_recall_per_image", 'Score': sm_recall_per_image(y_test, pred, threshold_pred)}
        except:
            record = {'Metric': "sm_recall_per_image", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    if all == 1:
        
        try:
            record = {'Metric': "np_recall_per_image", 'Score': np_p_r_f1(y_test, pred)[1]}
        except:
            record = {'Metric': "np_recall_per_image", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    if all == 1:
        
        try:
            record = {'Metric': "np_sensitivity_per_image", 'Score': np_specificity_np_sensitivity(y_test, pred)[1]}
        except:
            record = {'Metric': "np_sensitivity_per_image", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    if all == 1:
        
        try:
            record = {'Metric': "np_specificity_per_image", 'Score': np_specificity_np_sensitivity(y_test, pred)[0]}
        except:
            record = {'Metric': "np_specificity_per_image", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
    if all == 1:
        
        try:
            record = {'Metric': "np_object_precision_per_image", 'Score': np_object_p_r_f2(y_test, pred, threshold_pred)[0]}
        except:
            record = {'Metric': "np_object_precision_per_image", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    if all == 1:
        
        try:
            record = {'Metric': "np_object_recall_per_image", 'Score': np_object_p_r_f2(y_test, pred, threshold_pred)[1]}
        except:
            record = {'Metric': "np_object_recall_per_image", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    
    if all == 1:
        
        try:
            record = {'Metric': "np_object_f_per_image", 'Score': np_object_p_r_f2(y_test, pred, threshold_pred)[2]}
        except:
            record = {'Metric': "np_object_f_per_image", 'Score': -1}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)


    if all == 1:
        
        dice_arr, jaccard_arr, precision_arr, recall_arr, fpr_arr, fnr_arr, vs_arr, hd_arr, hd95_arr, msd_arr, mdsd_arr, stdsd_arr, tp_arr, tn_arr, fp_arr, fn_arr, false_omission_rate_arr, accuracy_arr = seg_metrics_fuc(y_test, pred)
        
        record = {'Metric': "dice_segmetric", 'Score': dice_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "jaccard_segmetric", 'Score': jaccard_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "precision_segmetric", 'Score': precision_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "recall_segmetric", 'Score': recall_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "fpr_segmetric", 'Score': fpr_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "fnr_segmetric", 'Score': fnr_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "vs_segmetric", 'Score': vs_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "hd_segmetric", 'Score': hd_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "hd95_segmetric", 'Score': hd95_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "msd_segmetric", 'Score': msd_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "mdsd_segmetric", 'Score': mdsd_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "stdsd_segmetric", 'Score': stdsd_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "tp_segmetric", 'Score': tp_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "tn_segmetric", 'Score': tn_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "fp_segmetric", 'Score': fp_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "fn_segmetric", 'Score': fn_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "false_omission_rate_segmetric", 'Score': false_omission_rate_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
        record = {'Metric': "accuracy_segmetric", 'Score': accuracy_arr}
        df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
        
    
    return df




import colorsys

def generate_light_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.3
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors

def generate_dark_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.3
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors

def highlight_values_with_colors(dataframe, column):
    unique_values = dataframe[column].unique()
    value_counts = dataframe[column].value_counts()
    dark_colors = generate_dark_colors(len(unique_values))
    light_colors = generate_light_colors(len(unique_values))
    
    color_map = {}
    for i, val in enumerate(unique_values):
        if value_counts[val] == 1:  # Unique value
            color_map[val] = (dark_colors[i % len(dark_colors)], 'white')  # Dark color with white text
        else:  # Repeated value
            color_map[val] = (light_colors[i % len(light_colors)], 'black')  # Light color with black text

    def apply_color(val):
        if val in color_map:
            bg_color, text_color = color_map[val]
            return f'background-color: {bg_color}; color: {text_color};'
        return ''

    return dataframe.style.applymap(apply_color, subset=[column])
    


import csv
import datetime


def format_time(seconds):
    """Format time in seconds to hh:mm:ss:ms format."""
    milliseconds = int((seconds - int(seconds)) * 1000)
    formatted_time = str(datetime.timedelta(seconds=int(seconds))) + f":{milliseconds:03d}"
    return formatted_time

def calc_time(epoch_times):    

    def format_time(seconds):
        """Format time in seconds to hh:mm:ss:ms format."""
        milliseconds = int((seconds - int(seconds)) * 1000)
        formatted_time = str(datetime.timedelta(seconds=int(seconds))) + f":{milliseconds:03d}"
        return formatted_time

    # Calculate the average epoch time excluding the first epoch
    average_time_seconds = sum(epoch_times[1:]) / len(epoch_times[1:])

    # Format the average time in hh:mm:ss:ms format
    average_time_formatted = format_time(average_time_seconds)

    # Calculate total training time
    total_training_time_seconds = sum(epoch_times)

    # Format the total training time in hh:mm:ss:ms format
    total_training_time_formatted = format_time(total_training_time_seconds)

    # List to store epoch data for CSV
    epoch_data = []

    # Process and print first 3 and last 3 epoch times in hh:mm:ss:ms format
    num_epochs = len(epoch_times)
    for i in range(num_epochs):
        # Print only first 3 and last 3 epochs
        if i < 3 or i >= num_epochs - 3:
            formatted_time = format_time(epoch_times[i])
            print(f"Epoch {i + 1} time: {formatted_time}")
            epoch_data.append({"Epoch": i + 1, "Time (hh:mm:ss:ms)": formatted_time})

    # Print the average epoch time excluding the first epoch
    print(f"Average epoch time (excluding first epoch): {average_time_seconds:.3f} seconds")
    print(f"Average epoch time (excluding first epoch): {average_time_formatted}")

    # Print the total training time
    print(f"Total training time: {total_training_time_seconds:.3f} seconds")
    print(f"Total training time: {total_training_time_formatted}")

    # Add summary data to epoch_data
    epoch_data.append({"Epoch": "Average (excl. 1st)", "Time (hh:mm:ss:ms)": average_time_formatted})
    epoch_data.append({"Epoch": "Total Training Time", "Time (hh:mm:ss:ms)": total_training_time_formatted})

    # Write the data to a CSV file
    csv_file = "logs/epoch_times.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Epoch", "Time (hh:mm:ss:ms)"])
        writer.writeheader()
        writer.writerows(epoch_data)

    print(f"Epoch times saved to {csv_file}")
    
    
    
    



##############################################################



from tensorflow.keras.models import Model
import sys
import pandas as pd
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime
from PIL import Image, ImageStat
import math
import numpy as np

def print_model_params(model):

    from keras_flops import get_flops
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.04} G")


    def get_model_memory_usage(model: Model, batch_size: int) -> float:
        mem_per_param = 4  # Each parameter is a 32-bit float, i.e., 4 bytes
        total_params = sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
        total_memory = total_params * mem_per_param
        
        # Memory required for the output
        output_memory = 0
        for output in model.outputs:
            output_shape = [batch_size] + output.get_shape().as_list()[1:]
            output_memory += np.prod(output_shape) * mem_per_param

        total_memory += output_memory
        return total_memory / (1024**2)  # Convert to megabytes


    print(f"Estimated memory usage: {round( get_model_memory_usage(model, 1) ,2)} MB")





#############################################################################################################################################

def create_pred_dirs():
    predictions_dir = 'predictions/'

    pred_path = predictions_dir + "pred/"
    pred_tta_path = predictions_dir + "pred_tta/"
    pred_avg_path = predictions_dir + "pred_avg/"
    pred_avg_tta_path = predictions_dir + "pred_avg_tta/"

    if os.path.exists(predictions_dir):
        shutil.rmtree(predictions_dir)

    os.makedirs(pred_path)
    os.makedirs(pred_tta_path)
    os.makedirs(pred_avg_path)
    os.makedirs(pred_avg_tta_path)



def save_pred(pred_dir_prefix, img1, img2, img3, img4, thresh, img_file_name, iou_img, iou_img_05, mean_pred_iou):
    
    
    img2 = img2[0]
    img3 = img3[0]
    img4 = img4[0]
    
    # Assuming you want each subplot to be close to 512 pixels wide, and given 6 subplots,
    # we adjust the figure size. This does not guarantee 512 pixels per image directly,
    # but it scales the figure to a size where subplots are larger.
    # DPI can be adjusted to fine-tune the output size.
    dpi = 300
    figsize_width = (1024 * 6) / dpi  # Calculate the width in inches for 6 images of 512 pixels each.
    fig, axs = plt.subplots(1, 6, figsize=(figsize_width, 1024/dpi), dpi=dpi)


    # fig, axs = plt.subplots(1, 6, figsize=(20, 5))

    axs[0].imshow(img1)
    axs[0].set_title('Image')
    axs[0].axis('off')

    axs[1].imshow(img2, cmap='gray')
    axs[1].set_title('GT')
    axs[1].axis('off')
    
    axs[2].imshow(img4, cmap='gray')
    axs[2].set_title('Pred @ 0.5')
    axs[2].axis('off')
    
    axs[3].imshow(img4, cmap='gray')
    axs[3].set_title('Overlay @ 0.5')
    axs[3].axis('off')
    c0 = axs[3].contour(img2.squeeze(), colors='r', levels=[0.9])

    axs[4].imshow(img3, cmap='gray')
    axs[4].set_title('Pred @ Best')
    axs[4].axis('off')

    axs[5].imshow(img3, cmap='gray')
    axs[5].set_title('Overlay @ Best')
    axs[5].axis('off')
    c1 = axs[5].contour(img2.squeeze(), colors='r', levels=[0.9])

    image_shape = img1.shape

    textstr = f'File: {img_file_name}\nImage Shape: {image_shape}\nIOU {iou_img_05}    Threshold: 0.5   |   IOU: {iou_img:.4f}   Threshold: {thresh}'
    plt.figtext(0.5, 0.01, textstr, wrap=True, horizontalalignment='center', fontsize=12)
    
    fig_str = pred_dir_prefix + os.path.split(img_file_name)[1] + ".jpg"
    plt.savefig(fig_str)
    
    df = pd.DataFrame(columns=['filename', 'iou@0.5', 'iou@best', 'mIoU'])
    record = {'filename':fig_str, 'iou@0.5': iou_img_05, 'iou@best': iou_img, 'mIoU': mean_pred_iou}
    df = pd.concat([df,pd.DataFrame([record])], ignore_index=True)
    df = df.round(4)
    df.to_csv(fig_str + ".csv")
    
    
    
    # plt.show()
    plt.close(fig)



def pred_full_best_iou(tta, control_sup, pred_index, pred_dir_prefix, model, image_path, mask_path, patch_img_size, patch_step_size, resize_img = False, resize_hw = [256,256], y_test_save = False):
    # print(mask_path)
    large_image = cv2.imread(image_path)
    large_mask = cv2.imread(mask_path)
    
    large_mask = large_mask[:,:,0]
    
    if (resize_img):
        large_image = cv2.resize(large_image, (resize_hw[0], resize_hw[1]))
        large_mask = cv2.resize(large_mask, (resize_hw[0], resize_hw[1]))

    print("---------------------------------------")
    
    print(image_path)
    print(mask_path)
    print(large_image.shape)
    print(large_mask.shape)
    
    patches = patchify(large_image, (patch_img_size, patch_img_size, 3), step=patch_step_size)

    patches_shape = patches.shape[0]

    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i,j,:,:]
            single_patch_norm = single_patch / 255
            single_patch_input=single_patch_norm
            
            if(tta):
                predictions_tta = []
                
                pred_original = model.predict(single_patch_input, verbose=0)
                if (control_sup == 1):
                    pred_original = pred_original[pred_index]
                else:
                    pred_original = pred_original
                pred_original = pred_original

                pred_lr = model.predict(np.fliplr(single_patch_input), verbose=0)
                if (control_sup == 1):
                    pred_lr = pred_lr[pred_index]
                else:
                    pred_lr = pred_lr
                pred_lr = np.fliplr(pred_lr)

                pred_ud = model.predict(np.flipud(single_patch_input), verbose=0)
                if (control_sup == 1):
                    pred_ud = pred_ud[pred_index]
                else:
                    pred_ud = pred_ud
                pred_ud = np.flipud(pred_ud)

                pred_lr_ud = model.predict(np.fliplr(np.flipud(single_patch_input)), verbose=0)
                if (control_sup == 1):
                    pred_lr_ud = pred_lr_ud[pred_index]
                else:
                    pred_lr_ud = pred_lr_ud
                pred_lr_ud = np.fliplr(np.flipud(pred_lr_ud))
                preds = (pred_original + pred_lr + pred_ud + pred_lr_ud) / 4

                predictions_tta.append(preds)
                
                single_patch_prediction = np.array(predictions_tta)
                

            else:
                single_patch_prediction = model.predict(single_patch_input, verbose = 0)
                if (control_sup == 1):
                    single_patch_prediction = single_patch_prediction[pred_index]
            
            predicted_patches.append(single_patch_prediction)

    predicted_patches = np.array(predicted_patches)

    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1],patches.shape[2], patch_img_size,patch_img_size) )

    predicted_patches_reshaped = predicted_patches_reshaped[:,:,0]

    reconstructed_image = unpatchify(predicted_patches_reshaped, [large_image.shape[0], large_image.shape[1]] )

    large_mask = np.expand_dims(large_mask, axis=0)
    reconstructed_image = np.expand_dims(reconstructed_image, axis=0)

    large_mask = np.expand_dims(large_mask, axis=-1)
    reconstructed_image = np.expand_dims(reconstructed_image, axis=-1)

    large_mask = np.where(large_mask>0, 1, 0)
    
    sm_iou = sm.metrics.IOUScore(per_image = True, threshold = 0.5)
    iou_img_05 = round( sm_iou(large_mask, reconstructed_image).numpy(), 4 )
    
    pred_mask_05 = np.where(reconstructed_image>0.5, 1, 0)

    thresholds = [0.5,0.6,0.7,0.8,0.9]
    pred_iou_thresh = []
    
    for thresh_i in range(len(thresholds)):
        sm_iou = sm.metrics.IOUScore(per_image = True, threshold = thresholds[thresh_i])
        iou_img = round( sm_iou(large_mask, reconstructed_image).numpy(), 4 )
        pred_iou_thresh.append(iou_img)
        
    
    index_of_max_value = pred_iou_thresh.index(max(pred_iou_thresh))
    
    max_thresh = thresholds[index_of_max_value]
    max_img_iou = pred_iou_thresh[index_of_max_value]
    
    
    mean_pred_iou = sum(pred_iou_thresh) / len(pred_iou_thresh)
    
    reconstructed_image = np.where(reconstructed_image>max_thresh, 1, 0)
    
    save_pred(pred_dir_prefix, large_image, large_mask, reconstructed_image, pred_mask_05, max_thresh, image_path, max_img_iou, iou_img_05, mean_pred_iou)
    
    print("max_thresh", max_thresh)
    print("max_img_iou", max_img_iou)
    print("iou_img_05", iou_img_05)
    print("mean_pred_iou", mean_pred_iou)
    
    
    
    print("---------------------------------------")
    
    if (y_test_save == False):
        large_mask = 0
    
    return large_mask ,reconstructed_image, pred_mask_05


def pred_full(tta, control_sup, pred_index, pred_dir_prefix, model, image_directory, mask_directory, patch_img_size, patch_step_size, resize_img, resize_hw, y_test_save):
    
    images = sorted(os.listdir(image_directory))
    print(images)
    print(len(images))

    masks = sorted(os.listdir(mask_directory))
    print(masks)
    print(len(masks))
    
    preds_reconstructed_array = []
    preds_reconstructed_05_array = []
    y_tests_array = []
    
    for i in range(len(images)):
        y_tests_, preds_, preds_05_ = pred_full_best_iou(tta, control_sup, pred_index, pred_dir_prefix, model, image_directory + images[i], mask_directory + masks[i], patch_img_size, patch_step_size, resize_img, resize_hw, y_test_save)
        preds_reconstructed_array.append(preds_[0])
        preds_reconstructed_05_array.append(preds_05_[0])
        if (y_test_save):
            y_tests_array.append(y_tests_[0])
    
    preds_reconstructed_array = np.array(preds_reconstructed_array)
    preds_reconstructed_05_array = np.array(preds_reconstructed_05_array)
    if (y_test_save):
        y_tests_array = np.array(y_tests_array)
    
    print(preds_reconstructed_array.shape)
    print(preds_reconstructed_05_array.shape)
    if (y_test_save):
        print(y_tests_array.shape)
    
    np.save(pred_dir_prefix + "preds_array.npy", preds_reconstructed_array)
    np.save(pred_dir_prefix + "preds_array_05.npy", preds_reconstructed_05_array)
    if (y_test_save):
        np.save(pred_dir_prefix + "y_tests_array.npy", y_tests_array)
    


def convert_to_same_size(imgs_array, size = [256,256]):
    imgs_array_temp = []

    for i in range(imgs_array.shape[0]):
        
        image = np.array(imgs_array[i]).astype(np.uint8)
        img = cv2.resize(image, (size[0],size[1]) )
        imgs_array_temp.append(img)

    imgs_array_temp = np.array(imgs_array_temp)

    return imgs_array_temp





####################################################
###----------------------------------------------###
###----------------------------------------------###
###----------------------------------------------###
###----------------------------------------------###
###----------------------------------------------###
####################################################


def eval_tta_patches(model, X_test, y_test, pred_index, threshold_pred, training, eval_results_path, best_epoch_str, model_clear_and_build, path_1):
    
    style_df_list = []

    prediction_times = []
    all_predictions = []

    for i in tqdm( range(len(X_test)) ):
        start_time = time.time()
        pred = model.predict(X_test[i:i+1], batch_size=1, verbose=0)
        end_time = time.time()
        
        if pred_index != -1:
            pred = pred[pred_index]

        pred = np.squeeze(pred, axis=0)
        
        prediction_time = end_time - start_time
        prediction_times.append(prediction_time)
        all_predictions.append(pred)

        # if i < 5:
        #     print(f"Prediction time for image {i+1}: {prediction_time} seconds")

    average_prediction_time = sum(prediction_times) / len(prediction_times)
    print(f"Average prediction time: {round(average_prediction_time,4)} seconds")

    pred = all_predictions

    # Saving the data to CSV
    df = pd.DataFrame({'Prediction_Time': prediction_times})
    df = df.round(4)
    df.to_csv('logs/prediction_times.csv', index=False)

    df = pd.DataFrame({'Average Prediction_Time': [average_prediction_time]})
    df = df.round(4)
    df.to_csv('logs/avg_prediction_time.csv', index=False)


    pred = np.array(pred)
    pred = pred.astype(np.float32)
    print("Prediction Array Details:  ")
    print(pred.shape)
    print('Min: %.3f, Max: %.3f' % (pred.min(), pred.max()))
    print(np.unique(pred))



    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)

    df.to_csv(eval_results_path + "all_metrics_" + best_epoch_str + ".csv")

    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 1 all_metrics_")
    print("---------------------------------------------------------------")



    image_dataset = X_test

    #Now that we know the transformations are working, let us extend to all predictions
    predictions = []
    for image in tqdm( image_dataset ):
        
        pred_original = model.predict(np.expand_dims(image, axis=0), verbose=0)
        if pred_index != -1:
            pred_original = pred_original[pred_index][0]
        else:
            pred_original = pred_original[0]
        pred_original = pred_original
        
        
        pred_lr = model.predict(np.expand_dims(np.fliplr(image), axis=0), verbose=0)
        if pred_index != -1:
            pred_lr = pred_lr[pred_index][0]
        else:
            pred_lr = pred_lr[0]
        pred_lr = np.fliplr(pred_lr)
        
        pred_ud = model.predict(np.expand_dims(np.flipud(image), axis=0), verbose=0)
        if pred_index != -1:
            pred_ud = pred_ud[pred_index][0]
        else:
            pred_ud = pred_ud[0]
        pred_ud = np.flipud(pred_ud)
        
        pred_lr_ud = model.predict(np.expand_dims(np.fliplr(np.flipud(image)), axis=0), verbose=0)
        if pred_index != -1:
            pred_lr_ud = pred_lr_ud[pred_index][0]
        else:
            pred_lr_ud = pred_lr_ud[0]
        pred_lr_ud = np.fliplr(np.flipud(pred_lr_ud))
        preds = (pred_original + pred_lr + pred_ud + pred_lr_ud) / 4
        
        predictions.append(preds)


    predictions = np.array(predictions)

    # print(predictions.shape)

    threshold = 0.5
    predictions_th = (predictions > threshold).astype(np.float32)

    pred = predictions_th


    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)
    df.to_csv(eval_results_path + "all_metrics_tta_" + best_epoch_str + ".csv")
    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 2 all_metrics_tta_")
    print("---------------------------------------------------------------")

    return style_df_list





##weights_folder_path = "results/"
def eval_tta_swa_patches(model, X_test, y_test, pred_index, threshold_pred, training, eval_results_path, best_epoch_str, model_clear_and_build, path_1, weights_folder_path = "results/"):
    
    style_df_list = []

    ###################### Ensembling Ensembling Ensembling Ensembling Ensembling Ensembling Ensembling Ensembling Ensembling Ensembling ######################

    print("SWA Processing Patches:")
    df = pd.read_csv("eval_results/all_eval.csv")
    max_sm_iou = df['sm_iou_score_per_image'].max()
    max_sm_iou = max_sm_iou * 100
    max_sm_iou = math.floor(max_sm_iou)
    print("Max IoU for SWA: ", max_sm_iou)

    def calc_epoch_list(df):
        filtered_df = df[df['sm_iou_score_per_image'] >= (max_sm_iou / 100)]
        selected_df = filtered_df[['Epoch', 'sm_iou_score_per_image']]
        sorted_df = selected_df.sort_values(by='sm_iou_score_per_image', ascending=False)
        epoch_list = sorted_df['Epoch'].tolist()
        
        epoch_list_len = len(epoch_list)
        if (epoch_list_len > 1 ):
            return epoch_list
        else:
            return epoch_list
            # new_max_sm_iou = max_sm_iou - 1
            # print(new_max_sm_iou)
            # filtered_df = df[df['sm_iou_score_per_image'] >= (new_max_sm_iou / 100)]
            # selected_df = filtered_df[['Epoch', 'sm_iou_score_per_image']]
            # sorted_df = selected_df.sort_values(by='sm_iou_score_per_image', ascending=False)
            # epoch_list = sorted_df['Epoch'].tolist()
            # return epoch_list
        
    epoch_list = calc_epoch_list(df)
    print("SWA Epochs: ", len(epoch_list))
    # print(epoch_list)
    
    print("SWA Finish: ______________________________________________")
    
    df = pd.DataFrame({'Epochs for SWA': epoch_list})
    df.to_csv('eval_results/epochs_SWA.csv', index=False)

    dirs_ = epoch_list  # Assuming epoch_list is defined elsewhere
    avg_weights = []

    # Initialize model structure
    model_clear_and_build()

    # Load the first set of weights to get the structure
    model.load_weights(weights_folder_path + dirs_[0])
    initial_weights = model.get_weights()

    # Prepare a list to store the average weights
    avg_weights = [np.zeros_like(w) for w in initial_weights]

    for i, dir_ in enumerate(dirs_):
        # print(i)
        # Load model weights
        model_clear_and_build()
        model.load_weights(weights_folder_path + dir_)
        current_weights = model.get_weights()
        
        # Accumulate weights
        for j, weight in enumerate(current_weights):
            avg_weights[j] += weight

    # Divide by the number of models to get the average
    avg_weights = [w / len(dirs_) for w in avg_weights]

    # Re-initialize and set the averaged weights
    model_clear_and_build()
    model.set_weights(avg_weights)

    # Save the averaged weights
    model.save_weights(path_1 + 'last_weights_average.hdf5')

    model_clear_and_build()
    model.load_weights("results/last_weights_average.hdf5")



    prediction_times = []
    all_predictions = []

    for i in tqdm( range(len(X_test)) ):
        start_time = time.time()
        pred = model.predict(X_test[i:i+1], batch_size=1, verbose=0)
        end_time = time.time()
        
        if pred_index != -1:
            pred = pred[pred_index]

        pred = np.squeeze(pred, axis=0)
        
        prediction_time = end_time - start_time
        prediction_times.append(prediction_time)
        all_predictions.append(pred)

        # if i < 5:
        #     print(f"Prediction time for image {i+1}: {prediction_time} seconds")

    average_prediction_time = sum(prediction_times) / len(prediction_times)
    print(f"Average prediction time: {round(average_prediction_time,4)} seconds")

    pred = all_predictions


    # pred = model.predict(X_test, batch_size=1, verbose=1)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    # print(pred.shape)
    # print('Min: %.3f, Max: %.3f' % (pred.min(), pred.max()))
    # print(np.unique(pred))



    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)

    df.to_csv(eval_results_path + "all_metrics_avgW_" + best_epoch_str + ".csv")

    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 3 all_metrics_avgW_")
    print("---------------------------------------------------------------")



    image_dataset = X_test

    #Now that we know the transformations are working, let us extend to all predictions
    predictions = []
    for image in tqdm( image_dataset ):
        
        pred_original = model.predict(np.expand_dims(image, axis=0), verbose=0)
        if pred_index != -1:
            pred_original = pred_original[pred_index][0]
        else:
            pred_original = pred_original[0]
        pred_original = pred_original
        
        
        pred_lr = model.predict(np.expand_dims(np.fliplr(image), axis=0), verbose=0)
        if pred_index != -1:
            pred_lr = pred_lr[pred_index][0]
        else:
            pred_lr = pred_lr[0]
        pred_lr = np.fliplr(pred_lr)
        
        pred_ud = model.predict(np.expand_dims(np.flipud(image), axis=0), verbose=0)
        if pred_index != -1:
            pred_ud = pred_ud[pred_index][0]
        else:
            pred_ud = pred_ud[0]
        pred_ud = np.flipud(pred_ud)
        
        pred_lr_ud = model.predict(np.expand_dims(np.fliplr(np.flipud(image)), axis=0), verbose=0)
        if pred_index != -1:
            pred_lr_ud = pred_lr_ud[pred_index][0]
        else:
            pred_lr_ud = pred_lr_ud[0]
        pred_lr_ud = np.fliplr(np.flipud(pred_lr_ud))
        preds = (pred_original + pred_lr + pred_ud + pred_lr_ud) / 4
        
        predictions.append(preds)


    predictions = np.array(predictions)

    # print(predictions.shape)

    threshold = 0.5
    predictions_th = (predictions > threshold).astype(np.float32)
    # predictions_th = np.where(predictions>threshold, 1, 0)
    predictions_th = (predictions > threshold).astype(np.float32)
    # predictions_th = np.expand_dims(predictions_th, axis=3)


    pred = predictions_th

    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)
    df.to_csv(eval_results_path + "all_metrics_tta_avgW_" + best_epoch_str + ".csv")
    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 4 all_metrics_tta_avgW_")
    print("---------------------------------------------------------------")

    return style_df_list





#########################################
#########################################
#########################################
#########################################
#########################################



def evaluation_full(control_sup, pred_index, model, image_full_test_directory, mask_full_test_directory, patch_img_size, patch_step_size, resize_img, resize_hw, resize_full_images, eval_results_path, threshold_pred, model_clear_and_build):

    style_df_list = []

    pred_dir_prefix = "predictions/pred/"
    tta = False
    pred_full(tta, control_sup, pred_index, pred_dir_prefix, model, image_full_test_directory, mask_full_test_directory, patch_img_size, patch_step_size, resize_img, resize_hw, y_test_save = True)

    y_test = np.load( "predictions/pred/" + "y_tests_array.npy", allow_pickle=True)
    
    if(resize_full_images):
        y_test = convert_to_same_size(y_test, size = [256,256])

    print(y_test.shape)
    print('Min: %.3f, Max: %.3f' % (y_test.min(), y_test.max()))
    print(np.unique(y_test))

    pred = np.load(pred_dir_prefix + "preds_array.npy", allow_pickle=True)

    if(resize_full_images):
        pred = convert_to_same_size(pred, size = [256,256])

    print(pred.shape)
    print('Min: %.3f, Max: %.3f' % (pred.min(), pred.max()))
    print(np.unique(pred))

    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)

    df.to_csv(eval_results_path + "all_metrics_full_pred"  + ".csv")

    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 1 all_metrics_full_pred")

    pred = np.load(pred_dir_prefix + "preds_array_05.npy", allow_pickle=True)

    if(resize_full_images):
        pred = convert_to_same_size(pred, size = [256,256])

    print(pred.shape)
    print('Min: %.3f, Max: %.3f' % (pred.min(), pred.max()))
    print(np.unique(pred))

    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)

    df.to_csv(eval_results_path + "all_metrics_full_pred_05"  + ".csv")

    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 2 all_metrics_full_pred_05")

    pred_dir_prefix = "predictions/pred_tta/"
    tta = True
    pred_full(tta, control_sup, pred_index, pred_dir_prefix, model, image_full_test_directory, mask_full_test_directory, patch_img_size, patch_step_size, resize_img, resize_hw, y_test_save = False)

    pred = np.load(pred_dir_prefix + "preds_array.npy", allow_pickle=True)

    if(resize_full_images):
        pred = convert_to_same_size(pred, size = [256,256])

    print(pred.shape)
    print('Min: %.3f, Max: %.3f' % (pred.min(), pred.max()))
    print(np.unique(pred))

    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)

    df.to_csv(eval_results_path + "all_metrics_full_pred_tta"  + ".csv")

    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 3 all_metrics_full_pred_tta")

    pred = np.load(pred_dir_prefix + "preds_array_05.npy", allow_pickle=True)

    if(resize_full_images):
        pred = convert_to_same_size(pred, size = [256,256])

    print(pred.shape)
    print('Min: %.3f, Max: %.3f' % (pred.min(), pred.max()))
    print(np.unique(pred))

    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)
    
    df.to_csv(eval_results_path + "all_metrics_full_pred_05_tta"  + ".csv")

    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 4 all_metrics_full_pred_05_tta")

    return style_df_list



def evaluation_full_Swa(control_sup, pred_index, model, image_full_test_directory, mask_full_test_directory, patch_img_size, patch_step_size, resize_img, resize_hw, resize_full_images, eval_results_path, threshold_pred, model_clear_and_build):

    style_df_list = []

    pred_dir_prefix = "predictions/pred/"
    tta = False
    pred_full(tta, control_sup, pred_index, pred_dir_prefix, model, image_full_test_directory, mask_full_test_directory, patch_img_size, patch_step_size, resize_img, resize_hw, y_test_save = True)

    y_test = np.load("predictions/pred/" +  "y_tests_array.npy", allow_pickle=True)

    if(resize_full_images):
        y_test = convert_to_same_size(y_test, size = [256, 256])

    print(y_test.shape)
    print('Min: %.3f, Max: %.3f' % (y_test.min(), y_test.max()))
    print(np.unique(y_test))
    
    model_clear_and_build()
    model.load_weights("results/last_weights_average.hdf5")

    pred_dir_prefix = "predictions/pred_avg/"
    tta = False
    pred_full(tta, control_sup, pred_index, pred_dir_prefix, model, image_full_test_directory, mask_full_test_directory, patch_img_size, patch_step_size, resize_img, resize_hw, y_test_save = False)

    pred = np.load(pred_dir_prefix + "preds_array.npy", allow_pickle=True)

    if(resize_full_images):
        pred = convert_to_same_size(pred, size = [256, 256])

    print(pred.shape)
    print('Min: %.3f, Max: %.3f' % (pred.min(), pred.max()))
    print(np.unique(pred))

    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)

    df.to_csv(eval_results_path + "all_metrics_full_pred_avgW"  + ".csv")

    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 5 all_metrics_full_pred_avgW")

    pred = np.load(pred_dir_prefix + "preds_array_05.npy", allow_pickle=True)

    if(resize_full_images):
        pred = convert_to_same_size(pred, size = [256, 256])

    print(pred.shape)
    print('Min: %.3f, Max: %.3f' % (pred.min(), pred.max()))
    print(np.unique(pred))

    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)

    df.to_csv(eval_results_path + "all_metrics_full_pred_05_avgW"  + ".csv")

    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 6 all_metrics_full_pred_05_avgW")

    pred_dir_prefix = "predictions/pred_avg_tta/"
    tta = True
    pred_full(tta, control_sup, pred_index, pred_dir_prefix, model, image_full_test_directory, mask_full_test_directory, patch_img_size, patch_step_size, resize_img, resize_hw, y_test_save = False)


    pred = np.load(pred_dir_prefix + "preds_array.npy", allow_pickle=True)

    if(resize_full_images):
        pred = convert_to_same_size(pred, size = [256, 256])

    print(pred.shape)
    print('Min: %.3f, Max: %.3f' % (pred.min(), pred.max()))
    print(np.unique(pred))

    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)

    df.to_csv(eval_results_path + "all_metrics_full_pred_tta_avgW"  + ".csv")

    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 7 all_metrics_full_pred_tta_avgW")

    pred = np.load(pred_dir_prefix + "preds_array_05.npy", allow_pickle=True)

    if(resize_full_images):
        pred = convert_to_same_size(pred, size = [256,256])

    print(pred.shape)
    print('Min: %.3f, Max: %.3f' % (pred.min(), pred.max()))
    print(np.unique(pred))

    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)

    df.to_csv(eval_results_path + "all_metrics_full_pred_05_tta_avgW"  + ".csv")

    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 8 all_metrics_full_pred_05_tta_avgW")
    
    return style_df_list



"""
#////////////////////////##////////////////////////##////////////////////////##////////////////////////##////////////////////////##////////////////////////#
#////////////////////////##////////////////////////##////////////////////////##////////////////////////##////////////////////////##////////////////////////#
#////////////////////////##////////////////////////##////////////////////////##////////////////////////##////////////////////////##////////////////////////#
#////////////////////////##////////////////////////##////////////////////////##////////////////////////##////////////////////////##////////////////////////#
#////////////////////////##////////////////////////##////////////////////////##////////////////////////##////////////////////////##////////////////////////#
#////////////////////////##////////////////////////##////////////////////////##////////////////////////##////////////////////////##////////////////////////#
#////////////////////////##////////////////////////##////////////////////////##////////////////////////##////////////////////////##////////////////////////#
"""



##weights_folder_path = "results/"
def evaluation_full_last_epochs(control_sup, pred_index, model, image_full_test_directory, mask_full_test_directory, patch_img_size, patch_step_size, resize_img, resize_hw, resize_full_images, eval_results_path, threshold_pred, model_clear_and_build, path_1, weights_folder_path = "results/"):

    ###################### Ensemble Last Last Last Last LastLastLast Last Ensemble Last Last Last Last LastLastLast Last Ensemble Last Last Last Last LastLastLast Last #####################
    
    print("SWA Last Processing:")

    data = pd.read_csv("logs/logs.csv")
    top_epochs = data.sort_values(by='loss', ascending=True).head(6)
    top_epochs = top_epochs + 1
    epoch_list = top_epochs['epoch'].tolist()
    epoch_list = [f"{num:04}" for num in epoch_list]
    epoch_list = [f"{item}.hdf5" for item in epoch_list]
    print("SWA Last Epochs: ", len(epoch_list))
    print(epoch_list)
    df = pd.DataFrame({'Epochs for SWA': epoch_list})
    df.to_csv('eval_results/epochs_SWA_Last.csv', index=False)

    dirs_ = epoch_list  # Assuming epoch_list is defined else where
    avg_weights = []

    # Initialize model structure
    model_clear_and_build()

    # Load the first set of weights to get the structure
    model.load_weights(weights_folder_path + dirs_[0])
    initial_weights = model.get_weights()

    # Prepare a list to store the average weights
    avg_weights = [np.zeros_like(w) for w in initial_weights]

    for i, dir_ in enumerate(dirs_):
        # print(i)
        # Load model weights
        model_clear_and_build()
        model.load_weights(weights_folder_path + dir_)
        current_weights = model.get_weights()
        
        # Accumulate weights
        for j, weight in enumerate(current_weights):
            avg_weights[j] += weight

    # Divide by the number of models to get the average
    avg_weights = [w / len(dirs_) for w in avg_weights]

    # Re-initialize and set the averaged weights
    model_clear_and_build()
    model.set_weights(avg_weights)

    # Save the averaged weights
    model.save_weights(path_1 + 'last_weights_average_last_epochs.hdf5')

    print("SWA Last Epoch Finish: ______________________________________________")

    ######################

    predictions_dir = 'predictions/'

    pred_avg_path_last_epochs = predictions_dir + "pred_avg_last_epochs/"
    pred_avg_tta_path_last_epochs = predictions_dir + "pred_avg_tta_last_epochs/"

    if os.path.exists(pred_avg_path_last_epochs):
        shutil.rmtree(pred_avg_path_last_epochs)
        shutil.rmtree(pred_avg_tta_path_last_epochs)

    os.makedirs(pred_avg_path_last_epochs)
    os.makedirs(pred_avg_tta_path_last_epochs)

    ######################

    style_df_list = []
    
    model_clear_and_build()
    model.load_weights("results/last_weights_average_last_epochs.hdf5")

    y_test = np.load("predictions/pred/" + "y_tests_array.npy", allow_pickle=True)
    
    if(resize_full_images):
        y_test = convert_to_same_size(y_test, size = [256,256])

    print(y_test.shape)
    print('Min: %.3f, Max: %.3f' % (y_test.min(), y_test.max()))
    print(np.unique(y_test))


    pred_dir_prefix = "predictions/pred_avg_last_epochs/"
    tta = False
    pred_full(tta, control_sup, pred_index, pred_dir_prefix, model, image_full_test_directory, mask_full_test_directory, patch_img_size, patch_step_size, resize_img, resize_hw, y_test_save = False)

    pred = np.load(pred_dir_prefix + "preds_array.npy", allow_pickle=True)

    if(resize_full_images):
        pred = convert_to_same_size(pred, size = [256,256])


    print(pred.shape)
    print('Min: %.3f, Max: %.3f' % (pred.min(), pred.max()))
    print(np.unique(pred))


    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)

    df.to_csv(eval_results_path + "all_metrics_full_pred_avgW_last_epochs"  + ".csv")

    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 1 all_metrics_full_pred_avgW_last_epochs")


    pred = np.load(pred_dir_prefix + "preds_array_05.npy", allow_pickle=True)

    if(resize_full_images):
        pred = convert_to_same_size(pred, size = [256,256])

    print(pred.shape)
    print('Min: %.3f, Max: %.3f' % (pred.min(), pred.max()))
    print(np.unique(pred))

    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)

    df.to_csv(eval_results_path + "all_metrics_full_pred_05_avgW_last_epochs"  + ".csv")

    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 2 all_metrics_full_pred_05_avgW_last_epochs")


    pred_dir_prefix = "predictions/pred_avg_tta_last_epochs/"
    tta = True
    pred_full(tta, control_sup, pred_index, pred_dir_prefix, model, image_full_test_directory, mask_full_test_directory, patch_img_size, patch_step_size, resize_img, resize_hw, y_test_save = False)


    pred = np.load(pred_dir_prefix + "preds_array.npy", allow_pickle=True)

    if(resize_full_images):
        pred = convert_to_same_size(pred, size = [256,256])

    print(pred.shape)
    print('Min: %.3f, Max: %.3f' % (pred.min(), pred.max()))
    print(np.unique(pred))


    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)

    df.to_csv(eval_results_path + "all_metrics_full_pred_tta_avgW_last_epochs"  + ".csv")

    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 3 all_metrics_full_pred_tta_avgW_last_epochs")


    pred = np.load(pred_dir_prefix + "preds_array_05.npy", allow_pickle=True)

    if(resize_full_images):
        pred = convert_to_same_size(pred, size = [256,256])

    print(pred.shape)
    print('Min: %.3f, Max: %.3f' % (pred.min(), pred.max()))
    print(np.unique(pred))


    df = eval_all_metrics(y_test, pred, all = 1, threshold_pred = threshold_pred)

    df.to_csv(eval_results_path + "all_metrics_full_pred_05_tta_avgW_last_epochs"  + ".csv")

    df = df.round(4)
    style_df = highlight_values_with_colors(df, 'Score')
    style_df_list.append(style_df)
    print("Done: 4 all_metrics_full_pred_05_tta_avgW_last_epochs")
    
    return style_df_list



#####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#####
#####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#####
#####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#####
#####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#####
#####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#####







