"""
A class for comparing two images and masks
"""
from .segmentation_metrics import intersection_over_union, iou_metrics
from .utils import gray_to_rgb
from ipywidgets import interact, Checkbox, FloatSlider, Dropdown, IntSlider
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

# change default matplotlib parameters
matplotlib.rcParams['figure.figsize'] = (10, 10)
matplotlib.rcParams['axes.titlesize'] = 18


class ObjectClass:
    def __init__(self, im, target_mask, pred_mask, verbose=1):
        """Class for comparing a target and prediction object mask.

        Params
        ------
        im : array-like
            the raw image, gray-scale image is supported but not RGB
        target_mask : array-like
            the object mask for the ground truth or the target
        pred_mask : array-like
            the object mask for the prediction
        """
        self.im = im
        self.target_mask = target_mask
        self.pred_mask = pred_mask

        # iou and intersection are arrays that are helpful to keep around for the rest of the methods
        iou, intersection = intersection_over_union(target_mask, pred_mask)
        self.iou = iou
        self.intersection = intersection

        # metrics can be calculated at different thresholds - keep these around in dict
        self.metrics = {'average': dict(
          f1=0, tp=0, fp=0, fn=0, official_score=0, precision=0, recall=0
        )}

        # calculate the metrics from threshold range of 0.05 to 0.95
        self._get_metrics()

        if verbose:
            self.report_metrics()  # report average scores

        # calculate masks
        self.combination_mask = self._calculate_masks()

    def _get_metrics(self):
        """Calculate the metrics for the prediction object mask using the Data Science Bowl 2018 approach."""
        average = self.metrics['average']  # using average as dict reference for cleaner code

        thresholds = np.arange(0.05, 0.96, 0.05)
        for threshold in thresholds:
            f1, tp, fp, fn, official_score, precision, recall = iou_metrics(self.iou, threshold)

            self.metrics[threshold] = dict(
                f1=f1, tp=tp, fp=fp, fn=fn, official_score=official_score, precision=precision, recall=recall
            )

            # add this threshold's values to the average entry
            average['f1'] += f1
            average['tp'] += tp
            average['fp'] += fp
            average['official_score'] += official_score
            average['precision'] += precision
            average['recall'] += recall

        # average the values
        average['f1'] /= len(thresholds)
        average['tp'] /= len(thresholds)
        average['fp'] /= len(thresholds)
        average['official_score'] /= len(thresholds)
        average['precision'] /= len(thresholds)
        average['recall'] /= len(thresholds)

    def report_metrics(self, threshold='average'):
        """Print out nicely the metrics at a given threshold - range from 0.05 to 0.95 in steps of 0.05 or the default
        is average which reports the average metrics (called by default on initiation)"""
        print(f'Reporting prediction mask metrics - compared to target mask for {threshold} threshold')

        for k, v in self.metrics[threshold].items():
            print('\t%s :  %.2f' % (k, v))

    def visualize(self, target=True, alpha=0.5):
        """Visualize the image and a single mask at the whole image level - all objects at once.

        If target is True then the target mask is plotted"""
        mask = self.target_mask if target else self.pred_mask
        label = 'Target Mask' if target else 'Predicted Mask'

        # plot overlayed
        plt.figure(figsize=(7, 7))
        plt.imshow(self.im, cmap='gray')
        plt.imshow(mask > 0, alpha=alpha)
        plt.title(f'Raw Image overlayed with {label}')
        plt.show()

        fig, ax = plt.subplots(ncols=2, figsize=(15, 7))
        plt.gca().axes.get_yaxis().set_visible(False)
        ax[0].imshow(self.im, cmap='gray')
        ax[0].xaxis.set_visible(False)
        ax[0].yaxis.set_visible(False)
        ax[0].set_title('Raw Image')
        ax[1].imshow(mask)
        ax[1].xaxis.set_visible(False)
        ax[1].yaxis.set_visible(False)
        ax[1].set_title(label)
        plt.show()

    def visualize_comparison(self, highlight='Raw'):
        """Visualize certain pixels of the image overlayed over the whole image. highlight parameter controls which
        pixels are highlighted - Raw just shows the raw image, Agreement highlights pixels as white if both target and
        predicted masks call that pixel a nuclei, FP stands for false positive and highlights pixels as red if only
        the predicted calls it a nuclei, and FN is highlighted pixels as blue if only the target mask calls it a nuclei.
        """
        # conver the grayscale image to rgb
        im = gray_to_rgb(self.im)

        # check the highlight condition
        if highlight == 'Agreement':
            im[(self.target_mask > 0) & (self.pred_mask > 0), :] = (255, 255, 255)
        elif highlight == 'FN':
            im[(self.target_mask > 0) & (self.pred_mask == 0), :] = (0, 0, 255)
        elif highlight == 'FP':
            im[(self.target_mask == 0) & (self.pred_mask > 0), :] = (255, 0, 0)
        elif highlight == 'All':
            im[self.target_mask > 0, :] = (0, 0, 255)
            im[self.pred_mask > 0, :] = (255, 0, 0)
            im[(self.target_mask > 0) & (self.pred_mask > 0), :] = (255, 255, 255)
        elif highlight == 'Target':
            im = self.target_mask > 0
        elif highlight == 'Prediction':
            im = self.pred_mask > 0

        print('White -> Agreement,  Blue -> False Negative,  Red -> False Positive')
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(im)
        plt.show()

    def visualize_objects(self, label, threshold=0.5):
        """Method for visualizing specific paired objects. This function shows all predicted objects that pass the iou
        threshold for a specific (label parameter) target object.

        Handling indices are weird here - remember the label is the label on the mask that corresponds to the image but
        the iou array (and match array) are index starting at 0 so you have to shift by 1.

        To be added -- label of 0 is not quite working yet, should display all nuclei at once..."""
        # threshold the iou
        match = self.iou > threshold

        # create a copy of the raw image in rgb
        im = gray_to_rgb(self.im)

        # each row in the match array represents a target object
        match_row = match[label - 1, :]

        # get prediction labels where there is intersection
        pred_labels = np.where(match_row)[0]

        # color codes as before - white for agreement, red for false positive, and blue for false negative
        target_mask = self.target_mask == label
        im[target_mask, :] = [0, 0, 255]

        # get indices where target mask is True
        row_i, col_i = np.where(target_mask)

        # color the matched predicted labels as red
        for pred_label in pred_labels:
            pred_mask = self.pred_mask == pred_label + 1
            im[pred_mask, :] = [255, 0, 0]
            im[pred_mask & target_mask, :] = [255, 255, 255]

            # get indices where pred mask is True and append row_i, col_i
            row_j, col_j = np.where(pred_mask)
            row_i, col_i = np.concatenate((row_i, row_j)), np.concatenate((col_i, col_j))

        # chop the region for a closer look =)
        # pad the edges of teh chopped regionpixels on each edge but threshold
        min_y, min_x, max_y, max_x = np.min(row_i), np.min(col_i), np.max(row_i), np.max(col_i)
        pad = 30
        min_y = max(0, min_y-pad)
        min_x = max(0, min_x-pad)
        max_x += pad
        max_y += pad
        chopped_im = im[min_y:max_y, min_x:max_x, :]

        # plot the images
        fig, ax = plt.subplots(ncols=2, figsize=(15, 7))
        ax[0].imshow(im)
        ax[0].xaxis.set_visible(False)
        ax[0].yaxis.set_visible(False)
        ax[1].imshow(chopped_im)
        ax[1].xaxis.set_visible(False)
        ax[1].yaxis.set_visible(False)
        plt.show()
        print(f'{len(pred_labels)} predicted object(s) overlap that pass iou threshold')

    def visualize_object_interact(self):
        _ = interact(
            self.visualize_objects,
            label=IntSlider(value=1, min=1, max=len(np.unique(self.target_mask))+1, continuous_update=False,
                            description='Nuclei Label', style={'description_width': 'initial'}),
            threshold=FloatSlider(value=0., min=0., max=1., continuous_update=False)
        )

    def visualize_comparison_interact(self):
        _ = interact(
            self.visualize_comparison,
            highlight=Dropdown(options=['Raw', 'Agreement', 'FN', 'FP', 'All', 'Target', 'Prediction'], value='Raw',
                               description='Type')
        )

    def _calculate_masks(self):
        """Calculate combination masks for comparing masks"""
        # combine the masks
        # color codes -
        # white - pixels that both masks call nuclei
        # red - pixels that was predicted as nuclei but are background in target mask
        # blue - pixels that were predicted as background but are nuclei in target mask
        combination_mask = np.zeros((self.im.shape[0], self.im.shape[1], 3), dtype=np.uint8)

        combination_mask[self.target_mask > 0, :] = (0, 0, 255)
        combination_mask[self.pred_mask > 0, :] = (255, 0, 0)
        combination_mask[(self.target_mask > 0) & (self.pred_mask > 0), :] = (255, 255, 255)
        return combination_mask

    def visualize_interact(self):
        """Wrapper around visualize for using in a Jupyter notebook widget."""
        _ = interact(
            self.visualize,
            target=Checkbox(value=True, description='Target mask?', indent=False),
            alpha=FloatSlider(value=0.5, min=0., max=1., step=0.05, continuous_update=False, description=
                              'Opacity of mask', style={'description_width': 'initial'})
        )
