# tecator-fda
A functional data analysis of the [Tecator meat sample data set](https://lib.stat.cmu.edu/datasets/tecator). Here, $X$ has shape (215, 100) and contains near infrared absorbance spectrum information for different wavelengths, while $y$ has shape (215, 1) and contains fat content for each meat sample. In particular, we compare OLS, ridge regression, and functional linear regression.

Disclaimer: parts, but not all, of this repository were created with the help of AI.

## Running this analysis on Linux

To run this analysis from scratch on a Linux system, assuming you have Python and Poetry installed, perform the following in the command line in the directory you wish to store this repository.
```
git clone git@github.com:nicholasgaubatz/tecator-fda.git
cd ../tectator-fda/
poetry install
```

Then, you can navigate to the `notebooks/` directory and run each notebook in sequence, starting with `notebooks/00_eda.ipynb` for an exploratory data analysis. Each notebook except for the first calls its own script in the `scripts/` directory to do the heavy lifting and simply pulls data from the `data/` directory for preesentation and discussion.

## Running this analysis on Windows 

In progress...