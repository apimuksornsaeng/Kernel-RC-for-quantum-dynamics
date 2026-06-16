# Sets the default math computation in numpy to not parallelise (might be MKL)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'    

import numpy as np
import csv
from time import time
from datagen.data_generate_dde import dde_rk45
from systems.ddes import mackeyglass
from utils.crossvalidation import CrossValidate
from utils.normalisation import normalise_arrays
from estimators.volt_funcs_mod import Volterra

if __name__ == "__main__":
            
    # Start wall timer
    start = time()
    
    SYSTEM = 'TFIM'
    N = 4
    hx = 8.0
    STATE = "Neel" # initial state: "Neel" or "Paramagnet"
    BC      = "periodic"  # 'open' or 'periodic'
    if BC == 'periodic':
        SYSTEM += f'_{BC}'
    real = 0
    mode = 'all' # 'all' or 'dynamics'

    # Load data from npy files
    if mode == 'dynamics':
        data_raw = np.expand_dims(np.load(f'data/{SYSTEM}_sz_dynamics_N{N}_hx{hx}_{STATE}_{real}.npy'), axis=1)
    elif mode == 'all':
        data_raw = np.load(f'data/{SYSTEM}_sz_all_N{N}_hx{hx}_{STATE}_{real}.npy').T
    # entropy_dynamics = np.load(f'data/{SYSTEM}_sent_N{N}_hx{hx}_{STATE}_{real}.npy')
    tgrid_raw = np.load(f'data/{SYSTEM}_tgrid.npy')

    print(f"Data shape is {data_raw.shape} and time grid shape is {tgrid_raw.shape}")

    delta_t = tgrid_raw[1] - tgrid_raw[0]

    # Define training and washout size
    t_train = 200
    ntrain = int(t_train / delta_t)
    t_wash = 50
    wash = int(t_wash / delta_t)
    data = data_raw[wash:] # Ensure data is long enough for training + washout
    tgrid = tgrid_raw[wash:]
    washout = 0
    selected_index = [N//2]

    # Construct training input and teacher, testing input and teacher
    training_input_orig = data[0:ntrain-1] 
    training_teacher_orig = data[1:ntrain]

    # Normalise training arrays if necessary
    normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig], norm_type=None)
    training_input, training_teacher = normalisation_output[0]

    # Define the range of parameters for which you want to cross validate over
    ld_coef_range = 1 - np.logspace(-7, 0, 23, endpoint=False) #np.linspace(0.1, 0.9, 9).round(1)
    tau_coef_range = np.logspace(-7, 0, 23, endpoint=False) #np.linspace(0.1, 0.9, 9).round(1)
    reg_range = np.logspace(-15, -1, 15)
    param_ranges = [ld_coef_range, tau_coef_range, reg_range]

    # Define additional input parameters
    param_add = [washout, selected_index]

    validation_train = 1000
    validation_test = 500
    validation_jump = 500

    validation_parameters = [validation_train, validation_test, validation_jump]

    # Instantiate CV, split dataset, crossvalidate in parallel
    CV = CrossValidate(validation_parameters=validation_parameters, validation_type="rolling", manage_remainder=True,
                       task="PathContinue", norm_type_in="ScaleL2Shift", error_type="meansquare", log_interval=100)
    cv_datasets = CV.split_data_to_folds(training_input, training_teacher)
    min_error, best_parameters_raw = CV.crossvalidate(Volterra, cv_datasets, param_ranges, param_add, 
                                                  num_processes=10, chunksize=1)      
    
    # ld_coef, tau_coef, reg_coef = best_parameters
    # ld_radius = ld_coef_range[1] - ld_coef_range[0]
    # tau_radius = tau_coef_range[1] - tau_coef_range[0]
    # reg_radius = reg_range[1] / reg_range[0]

    # ld_coef_range = np.linspace(0.1, 0.9, 9).round(1)
    # tau_coef_range = np.linspace(0.1, 0.9, 9).round(1)
    # reg_range = np.logspace(-15, -1, 15)


    # Print out the best paraeter and errors found
    print(f"Best parameters found are {best_parameters_raw} with error {min_error}")
    best_ld_coef, best_tau_coef, best_reg, _, _ = best_parameters_raw

    # convert best parameters to floats
    best_parameters = [float(best_ld_coef), float(best_tau_coef), float(best_reg)]

    # save best parameters and error to csv file
    
    # Store the filename in a variable to keep the code clean
    filename = f'cv_results/{SYSTEM}_{mode}_N{N}_hx{hx}_{STATE}_{real}_volterra.csv'

    # Convert validation_parameters to a string because CSV readers read data as strings
    val_str = str(validation_parameters) 
    is_duplicate = False

    # 1. Check if the file exists and look for duplicates
    file_exists = os.path.isfile(filename)

    if file_exists:
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                # Check if the row is not empty and the first column matches our validation set
                if row and row[0] == val_str:
                    is_duplicate = True
                    break

    # 2. Append the data if it is not a duplicate
    if not is_duplicate:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header ONLY if the file doesn't exist or is completely empty
            if not file_exists or os.path.getsize(filename) == 0:
                writer.writerow(['Validation', 'Best Parameters', 'Minimum Error'])
                
            writer.writerow([validation_parameters, best_parameters, min_error])
    else:
        # Optional: print a message or pass if you just want to fail silently
        # Overwrite the line with the same validation set (if you want to update results for the same validation set)
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)
            for i, row in enumerate(rows):
                if row and row[0] == val_str:
                    rows[i] = [val_str, best_parameters, min_error]  # Update the row with new results
                    break
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        # print(f"Validation set {validation_parameters} already exists. Skipping.")
    # Print amount of time taken to run cv
    print(f"Amount of time to run: {time() - start}")