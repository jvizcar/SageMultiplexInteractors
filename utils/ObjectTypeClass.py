"""
A class for visualizing dataset with both nuclei segmentation masks and labels for each of those cell to give it a type
of cell it is (read from a csv file).
"""
from pandas import read_csv, concat, merge
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import display
from math import ceil
import large_image
from imageio import imread
from .utils import normalize_image

from matplotlib import rcParams
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ipywidgets import interact, fixed, Dropdown, FloatSlider, IntSlider, SelectMultiple, Text, Checkbox
import ipywidgets as widgets

rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18

COLORS = (
    (0, 255, 0), (0, 0, 255), (255, 0, 0)
)

COLOR_NAMES = ['darkorange', 'cyan', 'darkred', 'fuchsia', 'darkgoldenrod', 'peru', 'deepskyblue']

COLORCODES = (
    'g', 'b', 'r'
)

COLOR_LABEL = (
    'Green', 'Blue', 'Red'
)


class ObjectTypeClass:
    def __init__(self, ometif_filepath, mask_filepath, region, csv_filepath, predictions=None, frame=0,
                 chname_filepath=None, pred_filepaths=None, markers_filepath=None, ft_importance_filepaths=None):
        """Initialize the object by providing an image and cell mask in numpy format and a filepath to the csv file
        with the labels.

        Modification - pass in the ome.tiff and mask for the entire ome.tiff but pass in a region dict to select a
        smaller region to visualize.

        Parameters
        ----------
        ometif_filepath : str
            filepath to ome.tif
        mask_filepath : str
            filepath to object (nuclei) mask for the ome.tif
        region : dict
            dictionary with left, top, width, and height keys for the small region to use for visualization
        csv_filepath : str
            filepath to csv file with object (nuclei) class labels and features
        chname_filepath : str
            filepath to txt file containing channel names in each new line - the line order matcher the frame order
            in the files
        pred_filepaths : dict
            keys are the label for csv file (value) containing predictions for each cell in csv_filepath. The named key
            is used to add a handle for each prediction file for the interactors.
        markers_filepath : str
            filepath to csv file containing the average marker values for each nuclei in the mask
        ft_importance_filepaths : dict
            keys are the label for csv file (value) containing feature importance for a prediction model on the nuclei
            label

        """
        # initialize the fields
        self.ts = large_image.getTileSource(ometif_filepath)
        self.mask_full = imread(mask_filepath)  # this is the mask for the entire ome.tif

        self.mask = self.mask_full[
                    region['top']:region['top']+region['height'], region['left']:region['left']+region['width']
                    ]

        # the image will be frame 0 which is assumed to be one of the DAPI channels
        self.kwargs = {'format': large_image.tilesource.TILE_FORMAT_NUMPY, 'frame': 0, 'region': region}
        im = normalize_image(self.ts.getRegion(**self.kwargs)[0][:, :, 0])
        self.im = im
        df = read_csv(csv_filepath)
        self.df_full = df.copy()
        self.forests = {}  # save the results of prediction labels using random forest classifiers

        # remove the rows that do not appear in this mask
        cell_ids = np.unique(self.mask)
        self.df = df[df.CellID.isin(cell_ids)].reset_index(drop=True)

        self.features = self.df.columns.tolist()
        del self.features[self.features.index('CellID')]
        del self.features[self.features.index('Label')]

        # extract the labels information in easy to access dictionaries
        self.labels = self.df_full.Label.unique().tolist()

        # delete the column with label names
        for label in self.labels:
            del self.features[self.features.index(label)]

        self.idx_map = {}
        self.label_map = {label: [] for label in self.labels}
        for label, cellid in zip(self.df.Label, self.df.CellID):
            self.idx_map[cellid] = label
            self.label_map[label].append(cellid)

        self.predictions = []
        if predictions is not None:
            for filepath in predictions:
                self.add_prediction(filepath)

        # calculate the mean marker intensity for each nuclei
        # self.marker_df = self.add_marker_feature()

        # add channel names dict or set it to None
        self.chnames = None
        self.read_chnames(chname_filepath)

        # add prediction files
        self.pred_dict = {}
        self.add_pred_filepath(pred_filepaths)

        if markers_filepath is None:
            self.markers_df = None
            self.markers = None
        else:
            # markers DataFrame
            self.markers_df = read_csv(markers_filepath)

            # get the list of marker names for interactors - removing the DAPI and background markers
            marker_list = []

            for marker in self.markers_df.columns.tolist():
                if marker not in ('Field_Row', 'Field_Col', 'CellID', 'Percent_Touching', 'Number_Neighbors') and\
                        'DAPI' not in marker and 'background' not in marker and marker not in self.features:
                    marker_list.append(marker)

            self.markers = marker_list

        # add the features importance
        self.ft_importance = {}
        if ft_importance_filepaths is not None:
            for ft_label, ft_filepath in ft_importance_filepaths.items():
                self.ft_importance[ft_label] = read_csv(ft_filepath)

    def add_pred_filepath(self, pred_filepaths):
        if pred_filepaths is not None:
            for pred_label, pred_filepath in pred_filepaths.items():
                self.pred_dict[pred_label] = read_csv(pred_filepath)

    def read_chnames(self, chname_filepath):
        """Read text file with channel names or set it to None

        chname_filepath : str
            filepath to text file with channel names in each line, the line number (indexed at 0) is assumed to match
            the frame index for that channel

        """
        if chname_filepath is None:
            self.chnames = None
        else:
            chname_dict = {}
            with open(chname_filepath) as fp:
                for i, line in enumerate(fp.readlines()):
                    chname_dict[line.strip()] = i

            self.chnames = chname_dict



    def _visualize(self, label, alpha):
        """Visualize the image and mask with selecting cell types."""
        overlay = np.zeros((self.im.shape[0], self.im.shape[1], 4), dtype=np.uint8)
        overlay[np.isin(self.mask, self.label_map[label])] = (255, 0, 0, 255)

        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(self.im, cmap='gray')
        ax.imshow(overlay, alpha=alpha)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.show()

    def visualize_interact(self):
        _ = interact(
            self._visualize,
            label=Dropdown(options=self.labels, value=self.labels[0], description='Label'),
            alpha=FloatSlider(value=0.5, min=0, max=1, step=0.1, description='Opacity', continuous_update=False)
        )

    def add_prediction(self, filepath):
        """Given a second csv file with different predictions for the cells - add to the object."""
        df = read_csv(filepath)
        idx_map = {}
        label_map = {label: [] for label in self.labels}
        for label, cellid in zip(df.Label, df.CellID):
            idx_map[cellid] = label
            label_map[label].append(cellid)

        self.predictions.append({'idx_map': idx_map, 'df': df, 'label_map': label_map})

    def _visualize_pred(self, label, idx=0, alpha=0.5):
        """Visualize the image and mask with selecting cell types and add a prediction overlay."""
        list1_as_set = set(self.label_map[label])
        intersection = list1_as_set.intersection(self.predictions[idx]['label_map'][label])

        mask = np.zeros((self.im.shape[0], self.im.shape[1], 4), dtype=np.uint8)
        mask[np.isin(self.mask, self.label_map[label])] = (255, 0, 0, 255)
        mask[np.isin(self.mask, list(intersection))] = (0, 255, 0, 255)

        cellids = set(np.unique(self.mask))
        labelids = cellids.intersection(self.label_map[label])
        trueids = labelids.intersection(self.predictions[idx]['label_map'][label])
        print('Accuracy %.2f percent' % (len(trueids) / len(labelids) * 100))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.im, cmap='gray')
        ax.imshow(mask, alpha=alpha)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.show()

    def visualize_pred_interact(self, idx):
        _ = interact(
            self._visualize_pred,
            idx=fixed(idx),
            label=Dropdown(options=self.labels, value=self.labels[0], description='Label'),
            alpha=FloatSlider(value=0.5, min=0, max=1, step=0.1, description='Opacity', continuous_update=False)
        )

    def visualize_features_interact(self):
        """Interactor for visualize features function"""
        labels = ['all'] + self.labels

        a = Dropdown(options=self.features, value=self.features[0], description='Feature')
        b = SelectMultiple(options=self.labels, value=self.labels, description='Labels')
        c = IntSlider(value=100, min=10, max=100, step=10, description='Num. Bins', continuous_update=False)
        d = Text(value='', placeholder='Location to save image', description='Save loc:')
        e = Checkbox(value=False, description='Save Figure', disabled=False)

        row1 = widgets.HBox([d, e])
        row2 = widgets.HBox([a, b, c])
        ui = widgets.VBox([row1, row2])

        out = widgets.interactive_output(self.visualize_features, {'feature': a, 'labels': b, 'bins': c, 'save_loc': d,
                                                                   'save': e})

        display(ui, out)

    def visualize_features(self, feature, labels, bins=100, save_loc='', save=False):
        """Visualize a feature value as heatmap on the nuclei with nuclei on top."""
        # create mask, with nuclei valued by their specific feature value
        mask = np.zeros((self.mask.shape[0], self.mask.shape[1]), dtype=np.float32)

        # loop through the unique cell values in image mask
        for value in np.unique(self.mask):
            if value != 0:
                # get the associated feature value for this cell
                row_index = self.df[self.df['CellID'] == value].index[0]
                feature_value = self.df.loc[row_index, feature]

                # use that feature value to fill in the array values for the cell
                mask[self.mask == value] = feature_value

        fig, ax = plt.subplots(ncols=3, figsize=(25, 7.5))

        ax[1].imshow(self.im, cmap='gray')
        ax[1].xaxis.set_visible(False)
        ax[1].yaxis.set_visible(False)

        label_mask = np.zeros((self.mask.shape[0], self.mask.shape[1], 4), dtype=np.uint8)

        for i, _label in enumerate(labels):
            color = list(COLORS[i])
            color.append(255 // 2)
            label_mask[np.isin(self.mask, self.label_map[_label])] = color

        ax[1].imshow(label_mask)
        im = ax[0].imshow(mask, cmap='YlOrRd')
        ax[0].xaxis.set_visible(False)
        ax[0].yaxis.set_visible(False)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # plot histogram of the feature for each group
        ns = []
        ft_value_list = []
        _min, _max = None, None
        for label in labels:
            group_rows = self.df_full[self.df_full.Label == label]
            ft_values = group_rows[feature].tolist()
            ft_value_list.append(ft_values)
            ns.append(len(ft_values))

            ft_min, ft_max = min(ft_values), max(ft_values)

            if _min is None or ft_min < _min:
                _min = ft_min
            if _max is None or ft_max > _max:
                _max = ft_max

        for i, value_list in enumerate(ft_value_list):
            _ = ax[2].hist(value_list, bins=bins, color=COLORCODES[i], alpha=0.5, range=(_min, _max+0.0001))

        legend = [f'{label} (n={n})' for label, n in zip(labels, ns)]
        ax[2].legend(legend, fontsize=18)
        ax[2].set_xlabel(feature, fontsize=18)
        ax[2].set_ylabel('Nuclei Count', fontsize=18)

        plt.suptitle(f'Feature - {feature}', fontsize=20)
        ax[0].set_title(f'Feature Heatmap')
        ax[1].set_title('Nuclei Class Mask')
        ax[2].set_title(f'Histogram of feature value by class (bins: {bins})')

        if save:
            plt.savefig(save_loc, bbox_inches='tight', dpi=300, pad_inches=0)
        plt.show()

    def visualize_feature_threshold(self, feature, min_val, max_val):
        """Visualize the overlay of cell type over the raw image - but threshold to only nuclei that meet a criteria
        for a feature. This allows you to get a better idea of what cell types comprise high and low value of features.
        """
        if min_val > max_val:
            print("The minimum value must be less then or equal to the max value")
        else:
            # threshold the nuclei so only the nuclei that fall within the thresholds remain
            if min_val == max_val:
                df = self.df[self.df[feature] == min_val]
            else:
                df = self.df[(self.df[feature] >= min_val) & (self.df[feature] <= max_val)]

            cell_ids = df.CellID.tolist()

            # create mask, with nuclei valued by their specific feature value
            # mask = np.zeros((self.mask.shape[0], self.mask.shape[1]), dtype=np.float32)
            mask = self.mask.copy()
            mask[~np.isin(mask, cell_ids)] = 0

            fig, ax = plt.subplots(figsize=(10,10))
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.imshow(self.mask, cmap='gray')

            label_mask = np.zeros((self.mask.shape[0], self.mask.shape[1], 4), dtype=np.uint8)

            for i, _label in enumerate(self.labels):
                color = list(COLORS[i])
                color.append(255 // 2)
                label_mask[np.isin(mask, self.label_map[_label])] = color
            ax.imshow(label_mask)
            plt.show()

            for i, _label in enumerate(self.labels):
                print(f'{_label} -> {COLOR_LABEL[i]}', end=';\t')

    def visualize_threshold_interact(self):
        _ = interact(
            self._intermediate,
            feature=Dropdown(options=self.features, value=self.features[0], description='Feature')
        )

    def _intermediate(self, feature):
        values = self.df[feature].tolist()
        min_val, max_val = min(values), max(values)

        if max_val <= 1:
            step = 0.05
        else:
            step = 1

        _ = interact(
            self.visualize_feature_threshold,
            feature=fixed(feature),
            min_val=FloatSlider(value=min_val, min=min_val, max=max_val, step=step, description='Min value',
                                continuous_update=False),
            max_val=FloatSlider(value=max_val, min=min_val, max=max_val, step=step, description='Max value',
                                continuous_update=False)
        )

    def predict_labels(self, name, features, forest_rs=64, split_rs=64, full=True, test_size=0.2, n_estimators=250):
        """Predict the labels using a set of features defined by the features list and added to prediction attribute
        key name. Also, calculate the feature importance. The prediction model used is random forest.

        The full parameter referes to using the data from the source csv file or only from the nuclei in the image
        region provided.
        """
        df = self.df_full if full else self.df

        # initate the random forest classifier
        forest = ExtraTreesClassifier(n_estimators=n_estimators, random_state=forest_rs)

        # extract the features array and label vector
        x = df[features]
        y = df['Label'].tolist()

        # split the feature array and label vector into training and testing parts
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, stratify=y,
                                                            random_state=split_rs)

        # fit the training data to random forest classifier (train)
        forest.fit(train_x, train_y)

        # get the feature importances
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]  # feature indices sorted from most important to least important

        # predict the probabilities on the test dataset
        y_prob = forest.predict_proba(test_x)

        # calculate the accuracy - print it out
        test_acc = forest.score(test_x, test_y)
        print(f'Accuracy on testing dataset for {name}: %.2f' % test_acc)

        # calculate the best accuracy (dummy classifier) from always classifying the most common class
        uniques = {}
        for _y in test_y:
            if _y not in uniques:
                uniques[_y] = 0
            uniques[_y] += 1
        max_count = max([count for count in uniques.values()])
        dummy_acc = max_count / len(test_y)
        print(f'\tThe dummy accuracy is %.2f' % dummy_acc)

        # one hot encode the labels for the testing dataset
        enc = OneHotEncoder(handle_unknown='ignore')
        onehot_labels = enc.fit_transform(np.reshape(test_y, (-1, 1))).toarray()

        # check that the forest and one hot encoder use the same order of labels
        if not np.all(enc.categories_ == forest.classes_):
            raise Exception('The one hot encoder and forest classifier are not using the same label order!')

        # calculate the axis for the ROC plot
        # source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(onehot_labels.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(onehot_labels[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(onehot_labels.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # save the results
        self.forests[name] = {
            'y_prob': y_prob,
            'importances': importances,
            'std': std,
            'indices': indices,
            'features': features,
            'y_test': test_y,
            'test_acc': test_acc,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }

    def plot_roc(self, names):
        """Plot the ROC curve for predicted label sets."""
        plt.figure(figsize=(10,10))
        lw = 2

        for i, name in enumerate(names):
            fpr = self.forests[name]['fpr']
            tpr = self.forests[name]['tpr']
            roc_auc = self.forests[name]['roc_auc']

            plt.plot(fpr[2], tpr[2], color=COLOR_NAMES[i],
                     lw=lw, label=f'{name} (AUC: %0.2f)' % roc_auc[2])

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right", fontsize=18)
        plt.show()

    def plot_class_roc(self, names, class_name):
        """Plot the ROC curve a list of predicted labels. Use the probabilities of the ground truth instead of the
        probabilities of the predictions."""
        plt.figure(figsize=(10, 10))
        lw = 2

        # grab the probabilities from the ground truth labels (as DataFrame with CellID column)
        prob_labels = self.df_full[['CellID'] + self.labels]

        for i, name in enumerate(names):
            # get the predicted labels
            pred_labels =  self.pred_dict[name]

            # merge the two dataframes using CellID column as restriction
            # * this keeps only the cells that have prediction labels
            df = merge(prob_labels, pred_labels, how='inner', on='CellID')

            # get the label column and one-hot encode (in this case these are the predictions)
            enc = OneHotEncoder(handle_unknown='ignore')
            onehot_labels = enc.fit_transform(np.reshape(df.Pred.tolist(), (-1, 1))).toarray()

            # category order used by encoded labels
            category_order = enc.categories_[0]

            # get the probabilities in order matching encoded labels
            y_prob = df[category_order].to_numpy()

            for idx, cat in enumerate(category_order):
                # this function only plots for one class at a time
                if cat == class_name:
                    fpr, tpr, _ = roc_curve(onehot_labels[:, idx], y_prob[:, idx])
                    roc_auc = auc(fpr, tpr)

                    # plot the roc for this category
                    plt.plot(fpr, tpr, color=COLOR_NAMES[i], lw=lw, label=f'{name} (AUC: %0.2f)' % roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.title(f'ROC for class {class_name}')
        plt.legend(loc="lower right", fontsize=18)
        plt.show()

    def interact_class_roc(self):
        """Use an interactor to select which ROC curves to display."""
        # get all possible predicted labels
        names = list(self.pred_dict.keys())

        # get the possible labels
        labels = self.labels

        _ = interact(
            self.plot_class_roc,
            names=SelectMultiple(options=names, value=names, description='Predicted Labels:',
                                 style={'description_width': 'initial'}),
            class_name=Dropdown(options=labels, value=labels[0], description='Class',
                                style={'description_width': 'initial'})
        )

    def interact_roc(self):
        """Use an interactor to select which ROC curves to display."""
        # get all possible predicted labels
        names = list(self.forests.keys())

        if len(names) == 0:
            raise Exception('You have not predicted any labels, call the predict_labels function!')

        _ = interact(
            self.plot_roc,
            names=SelectMultiple(options=names, value=names, description='Predicted Labels:',
                                 style={'description_width': 'initial'})
        )

    def plot_ft_importance(self, names, fig_height=5, max_features=100):
        """Plot the feature importance for each prediction using subplots."""
        # calculate the grid shape, limited to a max of 3 columns and as many rows as needed
        num = len(names)
        nrows = ceil(num / 3)
        ncols = ceil(num / nrows)
        if ncols == 3:
            w = 15
        elif ncols == 2:
            w = 10
        else:
            w = 5

        h = nrows * fig_height
        figsize = (w, h)
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

        try:
            ax = ax.ravel()
        except AttributeError:
            ax = [ax]

        for i, name in enumerate(names):
            # read the csv file
            df = self.ft_importance[name]

            # first line contains both the feature labels and the importances
            # make sure to sort them first
            series_sorted = df.iloc[0].sort_values(ascending=False)
            features = series_sorted.index.tolist()
            importances = series_sorted.values.tolist()

            if len(features) > max_features:
                features = features[:max_features]
                importances = importances[:max_features]

            ax[i].barh(features, importances, color="r", align="center")
            ax[i].set_xlabel('Feature Importance', fontsize=14)
            ax[i].set_yticklabels(features, fontsize=14)
            ax[i].set_ylim(ax[i].get_ylim()[::-1])
            ax[i].set_title(name, fontsize=18)
        plt.tight_layout()
        plt.show()

    def interact_ft_importance(self):
        """Interactor for displaying the feature importance for predicted labels."""
        # get all possible predicted labels
        names = list(self.ft_importance.keys())

        if len(names) == 0:
            raise Exception('You have not predicted any labels, call the predict_labels function!')

        _ = interact(
            self.plot_ft_importance,
            names=SelectMultiple(options=names, value=names, description='Predicted Labels:',
                                 style={'description_width': 'initial'}),
            fig_height=IntSlider(min=5, max=20, step=5, value=5, description='Subplot height:',
                                 style={'description_width': 'initial'}, continuous_update=False),
            max_features=IntSlider(min=3, max=100, step=1, value=100, description='Max number of features to plot:',
                                   style={'description_width': 'initial'}, continuous_update=False)
        )

    def plot_marker_ft_comparison(self, feature, markers, sample=1.0):
        """Plot scatter plot of a marker vs a feature, allows plotting more than one marker at a time using different
        colors.

        The sample parameter is a float from 0.0 to 1.0 that defines how many nuclei to sample from the total number
        of nuclei available to plot - plotting a smaller number of points makes comparisons easier to see."""

        # check that both dataframes matche in CellID column
        if not np.all(self.markers_df.CellID == self.df_full.CellID):
            raise Exception('The cell id columns of both dataframe do not match, not running function')

        # subset the markers dataframe by only the markers of interest and concatenate horizontally
        markers = list(markers)
        df = concat([self.df_full, self.markers_df[markers]], axis=1)

        # sample the nuclei
        if sample < 1.0:
            df = df.sample(frac=sample, random_state=64)

        # create the figure canvas
        fig, ax = plt.subplots(figsize=(10, 10))

        # plot the scatter of the feature vs each marker
        for i, marker in enumerate(markers):
            df.plot.scatter(x=feature, y=marker, c=COLOR_NAMES[i], ax=ax, alpha=0.2)

        ax.set_ylabel('Marker value', fontsize=18)
        ax.set_xlabel(f'Feature: {feature}', fontsize=18)
        ax.legend(markers, fontsize=14)
        ax.set_title('Comparison of nuclei feature and marker value', fontsize=20)
        plt.show()

    def interact_ft_vs_markers(self):
        """Interactor for plotting nuclei feature vs marker values."""
        _ = interact(
            self.plot_marker_ft_comparison,
            feature=Dropdown(options=self.features, value=self.features[0], description='Feature:'),
            markers=SelectMultiple(options=self.markers, value=[self.markers[0]], description='Select markers:',
                                   style={'description_width': 'initial'}),
            sample=FloatSlider(value=0.1, min=0.05, max=1.0, step=0.05, description='Frac. nuclei sampled:',
                               style={'description_width': 'initial'}, continuous_update=False)
        )

    def plot_heatmap_marker_ft_comparison(self, feature, markers, xbins, ybins, xmin_f=0., xmax_f=1., ymin_f=0.,
                                          ymax_f=1.):
        """Plot the results of comparing nuclei feature vs marker values using a heatmap approach to better visualize
        the density of points.

        xbins and ybins defines how many bins to use in each axis - the bins are of the same size which is determined
        by the min and max of those axis"""
        # check that both dataframes matche in CellID column
        if not np.all(self.markers_df.CellID == self.df_full.CellID):
            raise Exception('The cell id columns of both dataframe do not match, not running function')

        # subset the markers dataframe by only the markers of interest and concatenate horizontally
        markers = list(markers)
        df = concat([self.df_full, self.markers_df[markers]], axis=1)

        # determine the grid of subplots to show
        num = len(markers)
        nrows = ceil(num / 3)
        ncols = ceil(num / nrows)

        # the figsize should always look nice
        if ncols == 3:
            w = 15
        elif ncols == 2:
            w = 10
        else:
            w = 5
        h = nrows * 5
        figsize = (w, h)

        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, sharex='all', sharey='all')

        try:
            ax = ax.ravel()
        except AttributeError:
            ax = [ax]

        # create histograms for each marker
        for i, marker in enumerate(markers):
            # drop rows with infinite
            df_subset = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, marker])

            # get both axis
            x = df_subset[feature].tolist()
            y = df_subset[marker].tolist()

            x_max, y_max = max(x), max(y)
            _range = [[x_max * xmin_f, x_max * xmax_f], [y_max * ymin_f, y_max * ymax_f]]
        #
            im = ax[i].hist2d(x, y, (xbins, ybins), cmap=plt.cm.jet, range=_range)
            ax[i].set_xlabel(f'{feature}', fontsize=16)
            ax[i].set_ylabel(f'{marker}', fontsize=16)
            fig.colorbar(im[-1], ax=ax[i])

        fig.tight_layout()
        plt.show()

    def interact_hm_ft_vs_markers(self):
        """Interact for showing heatmaps for marker vs features."""
        a = Dropdown(options=self.features, value=self.features[0], description='Feature:')
        b = SelectMultiple(options=self.markers, value=[self.markers[0]], description='Select markers:',
                           style={'description_width': 'initial'})
        col1 = widgets.VBox([a, b])

        c = IntSlider(min=10, max=300, step=5, value=100, description='Horizontal num bins:',
                      style={'description_width': 'initial'}, continuous_update=False)
        d = IntSlider(min=10, max=300, step=5, value=100, description='Vertical num bins:',
                      style={'description_width': 'initial'}, continuous_update=False)
        col2 = widgets.VBox([c, d])

        e = FloatSlider(min=0., max=1., value=0., step=0.01, description='Min x:', orientation='vertical',
                        style={'description_width': 'initial'}, continuous_update=False)
        f = FloatSlider(min=0., max=1., value=1., step=0.01, description='Max x:', orientation='vertical',
                        style={'description_width': 'initial'}, continuous_update=False)
        g = FloatSlider(min=0., max=1., value=0., step=0.01, description='Min y:', orientation='vertical',
                        style={'description_width': 'initial'}, continuous_update=False)
        h = FloatSlider(min=0., max=1., value=1., step=0.01, description='Max y:', orientation='vertical',
                        style={'description_width': 'initial'}, continuous_update=False)

        ui = widgets.HBox([col1, col2, e, f, g, h])
        out = widgets.interactive_output(self.plot_heatmap_marker_ft_comparison,
                                         {'feature': a, 'markers': b, 'xbins': c, 'ybins': d, 'xmin_f': e, 'xmax_f': f,
                                          'ymin_f': g, 'ymax_f': h})
        display(ui, out)
