"""
A class for comparing two images and masks
"""
from .segmentation_metrics import intersection_over_union, iou_metrics
from .utils import gray_to_rgb, fix_segmentation_mask, normalize_image
import numpy as np
from IPython.display import display
from imageio import imread

from ipywidgets import interact, Checkbox, FloatSlider, Dropdown, IntSlider, Button, Text, fixed
import ipywidgets as widgets

from os import makedirs
from os.path import join

import matplotlib.pyplot as plt
import matplotlib
import large_image
from functools import partial

# change default matplotlib parameters
matplotlib.rcParams['figure.figsize'] = (10, 10)
matplotlib.rcParams['axes.titlesize'] = 18

DESC_STYLES = {'description_width': 'initial'}


def _create_button_text_display(default_value):
    # display button to save figures
    button = Button(description="Save Figure")
    text = Text(description='Save filename:', style=DESC_STYLES, value=default_value)

    return button, text


def _calculate_metrics(target_mask, pred_mask, threshold):
    # iou and intersection are arrays that are helpful to keep around for the rest of the methods
    iou, intersection = intersection_over_union(target_mask, pred_mask)

    if threshold == 'average':
        # metrics can be calculated at different thresholds - keep these around in dict
        metrics = dict(
          f1=0, tp=0, fp=0, fn=0, official_score=0, precision=0, recall=0
        )

        thresholds = np.arange(0.05, 0.96, 0.05)
        for th in thresholds:
            f1, tp, fp, fn, official_score, precision, recall = iou_metrics(iou, th)

            # add this threshold's values to the average entry
            metrics['f1'] += f1
            metrics['tp'] += tp
            metrics['fp'] += fp
            metrics['official_score'] += official_score
            metrics['precision'] += precision
            metrics['recall'] += recall

        # average the values
        metrics['f1'] /= len(thresholds)
        metrics['tp'] /= len(thresholds)
        metrics['fp'] /= len(thresholds)
        metrics['official_score'] /= len(thresholds)
        metrics['precision'] /= len(thresholds)
        metrics['recall'] /= len(thresholds)
    else:
        f1, tp, fp, fn, official_score, precision, recall = iou_metrics(iou, threshold)
        metrics = dict(
            f1=f1, tp=tp, fp=fp, fn=fn, official_score=official_score, precision=precision, recall=recall
        )

    return metrics


def _report_metrics(metrics):
    """Print out nicely the metrics at a given threshold - range from 0.05 to 0.95 in steps of 0.05 or the default
    is average which reports the average metrics (called by default on initiation)"""
    for k, v in metrics.items():
        print('\t%s :  %.2f' % (k, v))


class ObjectSegClass:
    def __init__(self, ometif_filepath, mask_filepaths, chnames_txt_file=None, region=None, region_ch=None,
                 save_dir='/data/Figures'):
        """Class for comparing a target and prediction object mask.

        Parameters
        ----------
        ometif_filepath : str
            filepath to ome.tif file
        mask_filepaths : dict
            filepaths to nuclei masks, the keys are the str handles used to select the masks in the interactors
        chnames_txt_file : str (default: None)
            filepath to text file containing the channel marker names
        region : dict (default: None)
            region to use for visualization, this should not be too large of a region (recommended to be less than 5k by
            5k pixels). If None is passed then a 1k by 1k region is used with left and top coordinates being 0,0.
        region_ch : str (default: None)
            the region is a small part of the image from a specified channel. This should ideally be the channel with
            the DAPI / DNA marker but you can specify the channel name here. If None then it will be the first channel
            that self.chnames.keys() gives.
        save_dir : str (default: '/data/Figures')
            the directory to save figures to from the interactors

        """
        # initialize the fields
        self.ts = large_image.getTileSource(ometif_filepath)

        # read the channel names from text file or directly from ome.tif metadata
        if chnames_txt_file is None:
            self.chnames = self.ts.getMetadata()['channelmap']  # this is the default channel map in metadata
        else:
            chname_dict = {}
            with open(chnames_txt_file) as fp:
                for i, line in enumerate(fp.readlines()):
                    chname_dict[line.strip()] = i
            self.chnames = chname_dict

        self.masks_dict = mask_filepaths

        # read the region for the channel provided or use the default option
        if region_ch is None:
            frame = self.chnames[list(self.chnames.keys())[0]]
        else:
            frame = self.chnames[region_ch]

        if region is None:
            region = {'left': 0, 'top': 0, 'width': 1000, 'height': 1000}
        self.region = region

        # read the region image
        self.kwargs = {'format': large_image.tilesource.TILE_FORMAT_NUMPY, 'frame': frame, 'region': region}
        self.region_im = self.ts.getRegion(**self.kwargs)[0][:, :, 0]

        # and read the associated regions for the mask
        self.region_masks = {}
        for mask_name, mask_filepath in self.masks_dict.items():
            mask = fix_segmentation_mask(imread(mask_filepath)[
                   region['top']:region['top']+region['height'],
                   region['left']:region['left']+region['width'],
                   ])
            self.region_masks[mask_name] = mask

        # create the save dir
        self.save_dir = save_dir
        makedirs(save_dir, exist_ok=True)

        # seed the metrics
        self.metrics = {}

    def on_button_clicked(self, b, text_widget=None, fig_to_save=None):
        filename = text_widget.value
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            pass
        else:
            filename = filename + '.png'

        savepath = join(self.save_dir, filename)
        try:
            fig_to_save.savefig(savepath, dpi=300, bbox_inches='tight', pad_inches=0.1)
            print(' ' * 100, end='\r')
            print(f'Figure saved to {savepath}', end='\r')
        except FileNotFoundError:
            print(f'Unable to save, please check filepath {savepath}!')

    def interact_ometif(self):
        """Interactor for visualizing the ome.tif file at low resolution and an associated region."""

        def _interact_ometif(frame, left, top, width, height):
            button, text = _create_button_text_display('ometif_region_seg.png')

            # get the region at 1.25x resolution for frame 0
            lr_im = self.ts.getRegion(format=large_image.tilesource.TILE_FORMAT_NUMPY, scale={'magnification': 1.25},
                                      frame=self.chnames[frame])[0][:, :, 0]

            fig, ax = plt.subplots(ncols=2, figsize=(15, 7))

            ax[0].imshow(lr_im, cmap='gray')
            ax[0].set_title(f'ome.tif channel {frame} at 1.25X magnification', fontsize=18)

            # get the region and reset the fields of regions....is this the best way to do it?
            region = {'left': left, 'top': top, 'width': width, 'height': height}
            self.region = region

            self.region_masks = {}
            for mask_name, mask_filepath in self.masks_dict.items():
                mask = fix_segmentation_mask(imread(mask_filepath)[
                       region['top']:region['top'] + region['height'],
                       region['left']:region['left'] + region['width'],
                       ])
                self.region_masks[mask_name] = mask

            reg_im = self.ts.getRegion(
                format=large_image.tilesource.TILE_FORMAT_NUMPY, region=region, frame=self.chnames[frame]
            )[0][:, :, 0]

            self.region_im = reg_im

            ax[1].imshow(reg_im, cmap='gray')
            ax[1].set_title('Image region at full magnification', fontsize=18)
            ax[1].xaxis.set_visible(False)
            ax[1].yaxis.set_visible(False)
            plt.suptitle('View OME.TIF at low magnification and select region to view at full mag', fontweight='bold',
                         fontsize=20)
            fig.tight_layout(rect=[0, 0, 1, .9])
            plt.show()

            button.on_click(partial(self.on_button_clicked, text_widget=text, fig_to_save=fig))
            display(widgets.HBox([text, button]))

        chnames = list(self.chnames.keys())
        a = Dropdown(options=chnames, value=chnames[0], description='Channel:', style=DESC_STYLES)
        b = IntSlider(min=0, max=self.ts.getMetadata()['sizeX'], value=self.region['left'],
                      description='Region left coord:', style=DESC_STYLES, continuous_update=False)
        c = IntSlider(min=0, max=self.ts.getMetadata()['sizeY'], value=self.region['top'],
                      description='Region top coord:', style=DESC_STYLES, continuous_update=False)
        d = IntSlider(min=100, max=5000, value=self.region['width'], description='Region width:', style=DESC_STYLES,
                      continuous_update=False)
        e = IntSlider(min=100, max=5000, value=self.region['height'], description='Region height:', style=DESC_STYLES,
                      continuous_update=False)

        row1 = widgets.HBox([a])
        row2 = widgets.HBox([b, d])
        row3 = widgets.HBox([c, e])
        ui = widgets.VBox([row1, row2, row3])

        out = widgets.interactive_output(_interact_ometif, {'frame': a, 'left': b, 'top': c, 'width': d, 'height': e})

        display(ui, out)

    def interact_visualize_mask(self):
        """Wrapper around visualize for using in a Jupyter notebook widget."""
        mask_names = list(self.masks_dict.keys())

        def _interact_visualize_mask(mask_name, alpha):
            """Visualize a mask on top of the image for the region only."""
            button, text = _create_button_text_display('mask_overlay.png')

            mask = self.region_masks[mask_name]
            im = self.region_im

            # convert the mask to RGB to make the non-nuclei areas transparent
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            mask_rgba[mask > 0, :] = (255, 0, 0, 255)

            # plot overlayed
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(im, cmap='gray')
            ax.imshow(mask_rgba, alpha=alpha)
            ax.set_title(f'Mask "{mask_name}" overlayed in region', fontsize=20, fontweight='bold')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            plt.show()

            button.on_click(partial(self.on_button_clicked, text_widget=text, fig_to_save=fig))
            display(widgets.HBox([text, button]))

        _ = interact(
            _interact_visualize_mask,
            mask_name=Dropdown(value=mask_names[0], options=mask_names, description='Select mask:', style=DESC_STYLES),
            alpha=FloatSlider(value=1., min=0., max=1., step=0.05, continuous_update=False,
                              description='Opacity of nuclei', style=DESC_STYLES)
        )

    def interact_metrics(self):
        """Interactor for calculating the metrics given two masks - use one as the target (ground truth) and the other
        as the prediction. Show some visualization as well are report the metrics."""

        mask_names = list(self.masks_dict.keys())

        if len(mask_names) == 1:
            pred_name = mask_names[0]
        else:
            pred_name = mask_names[1]

        def _interact_metrics(target_mask_name, pred_mask_name, threshold):
            button, text = _create_button_text_display('metrics_plot.png')

            target_mask = self.region_masks[target_mask_name]
            pred_mask = self.region_masks[pred_mask_name]

            # display both masks as at low res for the regions
            fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
            ax[0].imshow(target_mask, cmap='gray')
            ax[0].set_title(f'Target Mask: "{target_mask_name}"')
            ax[0].xaxis.set_visible(False)
            ax[0].yaxis.set_visible(False)
            ax[1].imshow(pred_mask, cmap='gray')
            ax[1].set_title(f'Prediction Mask: "{pred_mask_name}"')
            ax[1].xaxis.set_visible(False)
            ax[1].yaxis.set_visible(False)
            plt.show()

            # calculate the iou and intersection arrays using the regions....no option for doing this at full resolution
            # currently, too computationally intensive
            metrics = _calculate_metrics(target_mask, pred_mask, threshold)
            print(f'Reporting prediction mask metrics - compared to target mask for {threshold} threshold')
            _report_metrics(metrics)

            button.on_click(partial(self.on_button_clicked, text_widget=text, fig_to_save=fig))
            display(widgets.HBox([text, button]))

        thresholds = ['average', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        _ = interact(
            _interact_metrics,
            target_mask_name=Dropdown(value=mask_names[0], options=mask_names, description='Target mask:',
                                      style=DESC_STYLES),

            pred_mask_name=Dropdown(value=pred_name, options=mask_names, description='Prediction mask:',
                                    style=DESC_STYLES),
            threshold=Dropdown(value='average', options=thresholds, description='Metrics threshold:',
                               style=DESC_STYLES)
        )

    def interact_pixel_agreement(self):
        options = ['Raw', 'Agreement', 'FN', 'FP', 'All', 'Target', 'Prediction']
        pred_labels = list(self.region_masks.keys())
        target_label = pred_labels[0]

        if len(pred_labels) > 1:
            pred_label = pred_labels[1]
        else:
            pred_label = pred_labels[0]

        def _interact_pixel_agreement(target, pred, highlight):
            button, text = _create_button_text_display('nuclei_agreement.png')

            # convert the grayscale image to rgb
            im = gray_to_rgb(normalize_image(self.region_im))

            target_mask = self.region_masks[target].copy()
            pred_mask = self.region_masks[pred].copy()

            # check the highlight condition
            if highlight == 'Agreement':
                im[(target_mask > 0) & (pred_mask > 0), :] = (255, 255, 255)
            elif highlight == 'FN':
                im[(target_mask > 0) & (pred_mask == 0), :] = (0, 0, 255)
            elif highlight == 'FP':
                im[(target_mask == 0) & (pred_mask > 0), :] = (255, 0, 0)
            elif highlight == 'All':
                im[target_mask > 0, :] = (0, 0, 255)
                im[pred_mask > 0, :] = (255, 0, 0)
                im[(target_mask > 0) & (pred_mask > 0), :] = (255, 255, 255)
            elif highlight == 'Target':
                im = target_mask > 0
            elif highlight == 'Prediction':
                im = pred_mask > 0

            print('White -> Agreement,  Blue -> False Negative,  Red -> False Positive')
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_title(f'Nuclei Agreement\nTarget mask: {target}, Prediction mask: {pred}', fontsize=18)
            ax.imshow(im)
            plt.show()

            button.on_click(partial(self.on_button_clicked, text_widget=text, fig_to_save=fig))
            display(widgets.HBox([text, button]))

        _ = interact(
            _interact_pixel_agreement,
            target=Dropdown(options=pred_labels, value=target_label, description='Target mask:', style=DESC_STYLES),
            pred=Dropdown(options=pred_labels, value=pred_label, description='Prediction mask:', style=DESC_STYLES),
            highlight=Dropdown(options=options, value='Raw', description='Type:')
        )

    def interact_object(self):
        """Method for visualizing specific paired objects. This function shows all predicted objects that pass the iou
        threshold for a specific (label parameter) target object.

        Handling indices are weird here - remember the label is the label on the mask that corresponds to the image but
        the iou array (and match array) are index starting at 0 so you have to shift by 1.

        To be added -- label of 0 is not quite working yet, should display all nuclei at once..."""
        pred_labels = list(self.region_masks.keys())
        target_label = pred_labels[0]

        if len(pred_labels) > 1:
            pred_label = pred_labels[1]
        else:
            pred_label = pred_labels[0]

        def _intermediate(target):
            # based on the target mask get the labels for the rest of the interactors
            target_mask = self.region_masks[target]

            def _interact_object(target_mask, pred, label, threshold=0.5):
                button, text = _create_button_text_display('object_comparison.png')

                # get the target and prediction images
                pred_mask = self.region_masks[pred]

                # calculate the iou matrix
                iou = intersection_over_union(target_mask, pred_mask)[0]

                # threshold the iou
                match = iou > threshold

                # create a copy of the raw image in rgb
                im = gray_to_rgb(normalize_image(self.region_im.copy()))

                # each row in the match array represents a target object
                match_row = match[label - 1, :]

                # get prediction labels where there is intersection
                _pred_labels = np.where(match_row)[0]

                # color codes as before - white for agreement, red for false positive, and blue for false negative
                target_mask = target_mask == label
                im[target_mask, :] = [0, 0, 255]

                # get indices where target mask is True
                row_i, col_i = np.where(target_mask)

                # color the matched predicted labels as red
                for _pred_label in _pred_labels:
                    pred_mask = pred_mask == _pred_label + 1
                    im[pred_mask, :] = [255, 0, 0]
                    im[pred_mask & target_mask, :] = [255, 255, 255]

                    # get indices where pred mask is True and append row_i, col_i
                    row_j, col_j = np.where(pred_mask)
                    row_i, col_i = np.concatenate((row_i, row_j)), np.concatenate((col_i, col_j))

                # chop the region for a closer look =)
                # pad the edges of teh chopped regionpixels on each edge but threshold
                min_y, min_x, max_y, max_x = np.min(row_i), np.min(col_i), np.max(row_i), np.max(col_i)
                pad = 30
                min_y = max(0, min_y - pad)
                min_x = max(0, min_x - pad)
                max_x += pad
                max_y += pad
                chopped_im = im[min_y:max_y, min_x:max_x, :]

                print('White -> Agreement,  Blue -> False Negative,  Red -> False Positive')
                print(f'{len(pred_labels)} predicted object(s) overlap that pass iou threshold')

                # plot the images
                fig, ax = plt.subplots(ncols=2, figsize=(15, 7))
                ax[0].imshow(im)
                ax[0].xaxis.set_visible(False)
                ax[0].yaxis.set_visible(False)
                ax[1].imshow(chopped_im)
                ax[1].xaxis.set_visible(False)
                ax[1].yaxis.set_visible(False)
                plt.show()
                button.on_click(partial(self.on_button_clicked, text_widget=text, fig_to_save=fig))
                display(widgets.HBox([text, button]))

            _ = interact(
                _interact_object,
                target_mask=fixed(target_mask),
                pred=Dropdown(options=pred_labels, value=pred_label, description='Prediction mask:', style=DESC_STYLES),
                label=IntSlider(value=1, min=1, max=len(np.unique(target_mask)) - 1, continuous_update=False,
                                description='Nuclei Label', style={'description_width': 'initial'}),
                threshold=FloatSlider(value=0.5, min=0., max=1., continuous_update=False)
            )
        _ = interact(
            _intermediate,
            target=Dropdown(options=pred_labels, value=target_label, description='Target mask:', style=DESC_STYLES)
        )
