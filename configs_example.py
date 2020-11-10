"""
This is an example configs file used for running the interactor notebooks.

Note that you will need to move this file to your directory with images and rename it to configs.py. This file should
be at the root of the data directory that you will mount to /data in the container.

To run the example for the Lung3 dataset from the Sage hackathon 2020 challenge you can use the following links to
download the necessasry files. All files should be located in the directory that you will mount to /data in the Docker
container:

OMETIF_FILEPATH - https://www.synapse.org/#!Synapse:syn17778717
LABELMASK_FILEPATH - https://www.synapse.org/#!Synapse:syn21636364
LABELMASK_CSV - https://github.com/IAWG-CSBC-PSON/morpho-type/blob/master/data/Lung3.csv
CHNAMES_FILEPATH - https://github.com/IAWG-CSBC-PSON/morpho-type/blob/master/data/chNames.txt
MARKERS_FILEPATH - https://www.synapse.org/#!Synapse:syn19035696
PREDICTIONS - all lung3 files from https://github.com/IAWG-CSBC-PSON/morpho-type/tree/dc7efcdb75e864209d64acd9a1e0e2a3462e13d4/predictions/python_xgboost
FEATURE_IMPORTANCES - all lung3 files from https://github.com/IAWG-CSBC-PSON/morpho-type/tree/dc7efcdb75e864209d64acd9a1e0e2a3462e13d4/Yue/feature_impt

"""
# ----------------------------------------- Cell Type interactor paths ----------------------------------------------- #
# the interactors allow you to view a single ome.tif and associated files for that file such as a labeled mask and to
# use csv files with the results of analysis to plot ROC curves, feature importances, cell type calling, etc.
OMETIF_FILEPATH = '/data/path to ometif'
LABELMASK_FILEPATH = '/data/path to label mask file'

# contains index for each nuclei (based on value in mask) and morphological feautures
LABELMASK_CSV = '/data/path to csv file with morphological features for each nuclei'

# channel names - or set to None if you want these to be read from ome.tif metadata
CHNAMES_FILEPATH = '/data/path to text file with channel names'

# this file contains the average marker (channel in each round) value for each nuclei
MARKERS_FILEPATH = '/data/path to csv file with marker information'

# dictionary containing labels as keys (for interactor) and filepaths for values to csv file containing cell type
# predictions for each nuclei in the ome.tif. These are used to calculate ROC plots.
# set to None if you don't want to plot ROC plots.
PREDICTIONS = {
    'Description of file1': '/data/path to csv prediction file',
    'Description of file2': '/data/path to csv prediction file',
    'etc...': 'etc...'
}

# dictionary containing labels as keys (for interactors) and filepaths for values to csv files containing feature
# importances for each prediction model run. These are used to plot feature importance plots.
# set to None if you don't want to plot feature importance plots.
FEATURE_IMPORTANCES = {
    'Description of file1': '/data/path to csv feature importance file',
    'Description of file2': '/data/path to csv feature importance file',
    'etc...': 'etc...'
}

# this small region will be used to visualize part of the ome.tiff as an example in the interactors - ome.tiffs can be
# very large so trying to show the whole image at once would not work
REGION = {'left': 0, 'top': 0, 'width': 1000, 'height': 1000}


# --------------------------------------- Registration interactor paths ---------------------------------------------- #
# tiff_dirs is a dictionary where the keys are the label for each directory (used in interactors to select the dir of
# interest and the values is the path to the directory. If using this in Docker as intended then the path should start
# with /data/ followed by the path to your directory from location that you mounted as (understanding Docker mounting
# is required to understand the moutning paths).
tiff_dirs = {
    'Breast Cancer': '/data/breastCancer',
    'Healthy Breast': '/data/normalBreast'
}
