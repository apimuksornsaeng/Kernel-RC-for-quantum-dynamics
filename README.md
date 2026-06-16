# Kernel reservoir computing for quantum dynamics prediction

Repository for the project Infinite-dimensional next-generation reservoir computing.

Project structure:
* `datagen` : datasets and data generation files
* `estimators` : classes for all estimators used
* `images` : all images generated used in the paper
* `systems` : equations of used to generate data
* `utils` : cleaning, errors, normalisation, and plotting functions; crossvalidation class
* `cv_<estimator>_<system>.py` : crossvalidation for system and estimator
* `test_<estimator>_final.py` : training, testing, and image generation for all estimators

# Before running

For KRC for quantum system,
* `datagen/data_generate_TFIM.py` : a python file for generating quantum dynamics. You must run this file to generate the datasets first.
* run `cv_TFIM_volterra.py` to get the optimal Volterra kernel RC hyperparameters
* run `KRC_sandbox.ipynb` to visualize the prediction results and further analysis
* You can preliminarily visualize the quantum dynamics by `time_series.ipynb`

## Code requirements
Python 3.10.12 was used. Please see `requirements.txt` for the exact environment. 

## Authors 
This is joint work by
* Lyudmila Grigoryeva ([website](https://www.unisg.ch/en/university/about-us/organisation/detail/person-id/29fa04c7-1a51-43a6-a7ba-d300c0a661d9/), [scholar](https://scholar.google.com/citations?user=svYRWEMAAAAJ&hl=en))
* Hannah Lim Jing Ting ([scholar](https://scholar.google.com/citations?user=JOSbVKMAAAAJ&hl=en&oi=ao))
* Juan-Pablo Ortega ([website](https://juan-pablo-ortega.com), [scholar](https://scholar.google.com/citations?user=SoBQqSwAAAAJ&hl=en))

## Reference
Working paper versions: [ArXiv](https://arxiv.org/abs/2412.09800), [ResearchGate](https://www.researchgate.net/publication/386881122_Infinite-dimensional_next-generation_reservoir_computing)


