**Work in progress!**

# Using Machine Learning to Predict the Residual Load in Germany

With the help of deep learning, it is possible to outperform the prediction power of the public available forecasting from the transparency platform Entso-e. To accomplish this we use attention layers over the time series data, convolutions layers over the weather data, and finally dense layers over tabular data.

## Structure of the Repository

1. *01_Data:* There is a sub-folder for each data source. All data is publicly available and can be downloaded using the provided Jupyter Notebooks
2. *02_Support_Functions:* General functions designed for all the models
3. *03_Models:* Each model was saved in a Jupyter Notebook

## Setting up the environment

**Creating a virtual environment using Conda:** Make sure you have installed Anaconda. Pleas type the following in your terminal:

`$ conda create -n <name> python=3.8`

Activate the environment:

`$ conda activate <name>`

We provide all the requirements to gather the data and run all models in the requirements.txt file. To install in the new environment:

`$ pip install -r requirements.txt`
