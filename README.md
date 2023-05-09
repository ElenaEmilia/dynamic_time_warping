# PJ DS 2022 - K-Means with Fréchet Mean under Dynamic Time Warping for Data Imputation in Univariate Time Series

Missing or unknown data in time series has been a common drawback in the machine learning domain. Consecutively, missing values
would result in serious information loss if simply dropped from the dataset or ignored. To solve this pre-processing problem, there are
both machine learning approaches, and methods imported from statistical learning theory to impute the missing values. The aim of
this project is to analyze the missing data problem within univariate time series by comparing selected well-known methods with
k-means imputation under dynamic time warping using the Fréchet mean.

## Setup Instructions

### Conda Environment
For setting up the project, a [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment with all the necessary dependencies has to be created. If you haven't Conda installed already, please click the link above and follow the install guide for your operating system.

Execute these commands from the root directory of this project:
```
$ conda env create -n ds_pj -f conda/conda_env_pjds.yml
$ conda activate ds_pj
```

### TSLearn Modification
We extended the _TSLearn_ library with our implementation of the DTW zero cost heuristic. In order to finish the setup, two files from the now installed TSLearn library in the Conda environment have to be replaced with our modified version.

First, find the TSLearn directory of your just created Conda environment. To find out where your Conda environment is located, execute `conda env list` and follow the path. Enter the `tslearn` directory of your Conda environment: `ds_pj\Lib\site-packages\tslearn`.

Then, move the `src/tslearn_mod/metrics` directory from this repo into the TSLearn directory. This will replace two classes of TSLearn with our implementation.

## Project Structure

Following a short rundown of our project:

    ├── conda                   # Setup: Conda environment for the project
    ├── data                    # All UCR datasets. Used for experiments as our data
    ├── papers                  # Knowledge base for further reading
    ├── results                 # Scores and plots generated during the experiment will be stored here. Pre-computed results from our experiments are already added.
    ├── src                     # Tools and utilities
        ├── tslearn_mod         # Modified TSLearn library
        ├── utils
            └── load_ucr_data.py       # Loads UCR datasets
            └── preprocessing.py       # Contains data removal function and miscelleanous pre-processing functions
            └── visualization.py       # Various plot functions
        ├── dtw_mean.py         # DTW, DTW Zero Cost Heuristic, and SSG implementation
        ├── evaluation.py       # Evaluation pipeline
        ├── example.py          # Various examples showcasing functionalities of the project
        ├── experiment.py       # Reproduce our results. Experiment pipeline
        ├── imputation.py       # Contains all imputation methods
        └── k-means.py          # K-Means implementation for DTW K-Means

At the end of the project, we integrated our implementations of DTW Zero Cost Heuristic into the TSLearn library, as well as opted to use their DTW and SSG implementation instead of our own from the `dtw_mean.py` class. While the functionality is identical for both, TSLearn is heavily optimized for speed and memory usage, due to using [Cython](https://cython.org/)
and [Numba](https://numba.pydata.org/) instead of the standard Python compiler. This speeds up the computation significantly.


Some further elaborations on the stored results, these are stored in a JSON file of the following structure:
```
{
    "DATASET_NAME": {
        "IMPUTATION_METHOD": {
            "MISSING_SEQUENCE_LENGTH": {
                "r2": [SCORES_OVER_MISSING_RATES],
                "mse": [SCORES_OVER_MISSING_RATES],
            }
        }
    }
}
```

The Fréchet variances for the DTW K-Means runs are in the following format:
```
{
    "DATASET_NAME": {
        "CLUSTER_FACTOR": [FRECHET_VARIANCE_FOR_CLUSTERS]
    }
}
```

## Reproducibility
In order to reproduce our results, simply execute the `experiment.py` script **from the root directory**:
```
python src/experiment.py
```
This will generate all scores and plots after computation, which will be stored in the `/results` directory.

Based on our experience, sequential execution of experiment 1 (short time series) will take about 8 hours for computation. Experiment 2 (long time series) will take about 3 days. We recommend splitting the experiment into multiple parts to compute in parallel.

To get a run quickly done in order to observe our experimental and evaluation pipeline, we recommend commenting out experiment 2 from the main script, as well as executing experiment 1 on only the shortest time series.
Therefore, modify experiment 1 to take, for example, only _Chinatown_, _ECG200_, _ItalyPowerDemand_, _MoteStrain_, _SonyAIBORobotSurface1_ and _TwoLeadECG_. Note that the results won't be representative based on this small subset, but they give a good impression on the experimental design and process.

We also added our pre-computed results for experiment 1 and 2 to the results directory. If you want to generate plots for the pre-computed scores, execute the method `precomputed_experiment_1()` or `precomputed_experiment_2()` in this class.
## Examples

In `example.py`, we created a list of example showcasing different functionalities of this project. The examples can be executed using:
```
python src/example.py
```

## Authors and Acknowledgment
_Elena Kranz_\
kranz@campus.tu-berlin.de

_Saman Akbari_\
saman.akbari@campus.tu-berlin.de

_Efthimios-Enias Gojka_\
gojka@campus.tu-berlin.de

_Mats Salewski_\
m.salewski@campus.tu-berlin.de

_Anton Balthasar Jaekel_\
anton.b.jaekel@campus.tu-berlin.de

We would like to thank _David Schultz_ for his engagement and guiding throughout the semester, and therefore making this research possible.
