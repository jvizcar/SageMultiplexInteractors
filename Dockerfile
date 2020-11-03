# Pull from scipy jupyter base image
FROM jupyter/scipy-notebook:acb539921413

# install additional packages
RUN pip install girder_client
RUN python -m pip install histomicstk --find-links https://girder.github.io/large_image_wheels --prefer-binary

# switch to root user to create the directories to place code in and the data
USER root
USER mkdir /code & mkdir /data
USER chmod 777 -R /code & chmod 777 -R /data
WORKDIR /code

# swith back to jovyan user
USER jovyan

# install nbextensions for Jupyter
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable codefolding/main
RUN jupyter nbextension enable init_cell/main

# configure the notebook to display in 100% width
RUN mkdir /home/jovyan/.jupyter/custom
RUN echo '.container {width:100% !important;}' > /home/jovyan/.jupyter/custom/custom.css

# add the code of this repository to /code directory
ADD . /code

# open up permisssions to /code again
USER root
RUN chmod 777 -R /code
USER jovyan

# switch the working directory to notebook dir
WORKDIR /code/notebooks

# trust the notebooks
# jupyter trust path_to_notebook   (do for each)
