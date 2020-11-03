"""
A class for comparing multiple prediction masks to a target mask.
"""
from . import ObjectClass
import numpy as np
from .utils import gray_to_rgb
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider, fixed


class ObjectsClass:
    def __init__(self, im, target_mask, pred_masks, verbose=1):
        """Class for comparing a target and prediction object mask.

        Params
        ------
        im : array-like
            the raw image, gray-scale image is supported but not RGB
        target_mask : array-like
            the object mask for the ground truth or the target
        pred_masks : sequence
            sequence of object masks for the prediction methods
        """
        self.im = im
        self.target_mask = target_mask
        self.pred_masks = pred_masks

        # create comparisons objects
        self.pred_objects = []
        for i, pred_mask in enumerate(pred_masks):
            if verbose:
                print(f'\nPrediction Mask {i+1}')
                print('-'*100)
            self.pred_objects.append(ObjectClass(im, target_mask, pred_mask, verbose=verbose))

    def visualize_heatmap(self, threshold=0.5, alpha=255):
        """Visualize the target nuclei - highlighted for agreement"""
        step = 255 // len(self.pred_objects)

        # get iou at threshold
        iou_shape = self.pred_objects[0].iou.shape
        match = np.zeros(iou_shape[0], dtype=np.uint8)
        for pred_object in self.pred_objects:
            iou = pred_object.iou > threshold
            iou_sum = np.sum(iou, axis=1)
            match[iou_sum > 0] += 1
        match *= step  # scale to 255 as max

        # create copy of image to highlight the cells that were found and those that were not
        im = gray_to_rgb(self.im.copy())
        overlay = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.uint8)
        for i in range(len(match)):
            if match[i] > 0:
                overlay[self.target_mask == i+1] = (match[i], 0, 0, 255)

        ax = plt.subplots(figsize=(10, 10))[1]
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(im)
        ax.imshow(overlay, alpha=alpha)
        plt.show()

    def visualize_heatmap_interact(self):
        _ = interact(
            self.visualize_heatmap,
            threshold=FloatSlider(min=0., max=1., value=0.5, step=0.05, continuous_update=False),
            alpha=FloatSlider(min=0., max=1., step=0.05, value=1., continuous_update=False)
        )

