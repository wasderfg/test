# Code & Data for Submission 10279
This is an implementation of GDCF, and the work is submitted to AAAI2023.
# Environment
- Python 3.6.12
- PyTorch 1.6.0
- NumPy 1.19.1
- tqdm 4.51.0
# Dataset
The datasets are sampled and placed in data/ directory.
- **NYTaxi** is data from first 21 days of NYTaxi.

# Training
- Train the autoencoder: **python train_phase1.py** This will generate node representations for the whole dataset and save them under the directory output/phase1/results.
- Train the downstream task: **python train_phase2.py --task=od/o/i**
- Train model for travel time estimation: **python travel_time_estimation.py**

More hyperparameters can be viewed in the source code and adjusted with the running demand.