# edbt-2027

# MATLAB requirements 
We use the DVS saliency model (written in MATLAB). You will need your own package installation, which can be found [here](https://github.com/sandialabs/DVS). 

You should replace the file found at `dvs_release/textSaliency/textSaliency.m` with the `textSaliency.m` we provide in our repo; we had to slightly modify the original version for our purposes. 

Our code is tested in MATLAB R2022_b. You also need to install a MATLAB Engine for Python (version: 9.13.1) so you can call the MATLAB scripts through Python. 

More information can be found [here](https://pypi.org/project/matlabengine/9.13.1/) and [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html). 

# Data  
Due to storage constraints on Anonymous Github, we share our data using an anonymous OSF [link](https://osf.io/gfrxk/?view_only=09ba4d97978b476eac0075ac07872c73). Please download the files and unzip them in the folder where you cloned the repo. The paths are organized as follows:

### .../dataset_name/ 
- `saliency_data/`                        # aggregate saliency map information
  - `cumulative_saliency/`    # saliency maps of the original data stored as images across different configurations of point size & opacity
- `sampling_techniques/`
  - `evaluation_metrics/`                  
  - `cumulative_saliency/data`              # samples of different sizes 
  - `cumulative_saliency/images`            # saliency maps of the samples stored as images for all configurations of point size & opacity

# There are four notebooks to demonstrate the functionality of the perception-aware framework.
  1. data_prep -- for computing an aggregate saliency map and deriving perception weights.
  2. sampling -- for deriving samples using state-of-the-art sampling methods and computing the saliency maps needed for evaluation.
  3. evaluation -- for computing the evaluation metrics.
  4. appropaws -- for computing compressed data representations and approximate visualizations.

Sampling.ipynb is the main notebook for deriving samples for a dataset with computed perception weights.



  


