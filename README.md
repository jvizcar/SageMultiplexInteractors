# Sage Multiplex Image Interactors
10 Nov. 2020

Provides Jupyter notebooks that are aimed at providing interactive visualization of multiplex images themsevles and
analysis results on those images.

This notebook is meant to be run in a Docker container by mounting your data directory - data directory contains your
images, masks, and data csv files. Since all the paths are configured in a configs.py file it is possible to run the 
notebooks outside of a Docker container so long as you configure the paths appropriately.

To run with a Docker container use the following command which allows you to port-forward the Jupyter notebook port 
locally and to mount your data dir which should be mount to /data in the Docker container.

```docker run -it --rm -p####:8888 -v <data_dir>:/data jvizcar/sage_multiplex_interactors:latest```

Make sure to fill #### with the local port you want to port-forward to - the token password will be displayed in the 
terminal, copy paste this to access the notebook. <data_dir> should be the absolute path to your directory containing
all the images, image dirs, masks, and csv files.

Look into the configs_example.py file to understand the required files you will need to run the notebooks.