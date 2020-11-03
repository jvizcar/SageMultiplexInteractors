"""
Functions used in comparing segmentation masks.
"""
import numpy as np


def intersection_over_union(ground_truth, prediction):
    """Calculate the intersection over union between two object segmentation masks. For this to work appropriately the
    masks encode each object by making the pixels that make up that object with a unique int value. The encoding or
    int labels must be in increasing order, i.e. if there are two objects then the one object must use 1s as the label
    and the other must use 2s as the label. If you don't use increasing ints as the labels (with step size of 1) then
    the metrics predicted will be wrong.

    Source: https://github.com/carpenterlab/2019_caicedo_dsb/blob/master/evaluation.py

    Parameters
    ----------
    ground_truth : array-like
        ground truth object segmentation mask
    prediction : array-like
        prediction of object segmentation mask

    Return
    ------
    iou : array-like
        the intersection over union between the two masks for each paired objects (a pair is an object from the target
        mask and an object form the prediction mask)
    interesection : array-like
        intersection histogram between all target objects and prediction objects. The rows represent target objects and
        the columns are prediction objects.

    """
    # check that there is no negatives in the arrays
    assert np.all(ground_truth >= 0)
    assert np.all(prediction >= 0)

    # number of objects in each mask
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))

    # 2D histogram will provide counts above 0 for objects that have at least 1 pixel of overlap
    # in other words - the number of pixels that overlap
    h = np.histogram2d(ground_truth.flatten(), prediction.flatten(), bins=(true_objects, pred_objects))
    intersection = h[0]

    # area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]

    # calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection

    # exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]

    # compute intersection over Union
    union[union == 0] = 1e-9  # this should prevent errors if the union was 0
    iou = intersection / union

    return iou, intersection


def iou_metrics(iou, threshold=0.5):
    """Calculate various metrics on the intersection over union array at a given threshold.

    Parameters
    ----------
    iou : array-like
        the intersection over union array between two masks objects, output from intersection_over_union function
    threshold : float (default=0.5)
        0.0 to 1.0 threshold for IOU

    Returns
    -------
    f1 : float
        the F1 score
    TP : int
        number of true positive objects
    FP : int
        number of false positive objects
    official_score : float
        the official score given by the rules of the 2018 Data Science Bowl challenge
    precision : float
        the precision
    recall : float
        the recall

    """
    # threshold the IOU array
    matches = iou > threshold

    true_positives = np.sum(matches, axis=1) == 1  # correct objects
    false_positives = np.sum(matches, axis=0) == 0  # extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # missed objects

    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))

    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    official_score = tp / (tp + fp + fn + 1e-9)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    return f1, tp, fp, fn, official_score, precision, recall
