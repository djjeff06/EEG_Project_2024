# Author: Jefferson Dionisio
# Date: 2024-06-04

#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
import joblib
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import math
import random
from torch.optim.lr_scheduler import StepLR
import re
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc, make_scorer, roc_auc_score, average_precision_score, f1_score, mean_squared_error, mean_absolute_error

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')
        
    mne.set_log_level('WARNING')
    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    outcomes = list()
    cpcs = list()
    hours_list = list()
    recording_patient_dictionary = list()
    eeg_features = list()
    padding_masks = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))
            
        print("Processing patient#",i)
        current_eeg_features, current_hours, current_masks = get_features(data_folder, patient_ids[i])

        print(len(current_eeg_features), ' recordings found!')

        for j in range(len(current_hours)):
            hours_list.append(current_hours[j])
            recording_patient_dictionary.append(i)
            eeg_features.append(current_eeg_features[j])
            padding_masks.append(current_masks[j])

        # Extract labels.
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    outcomes = np.array(outcomes)
    cpcs = np.array(cpcs)
    hours_list = np.array(hours_list)
    recording_patient_dictionary = np.array(recording_patient_dictionary)
    eeg_features = np.array(eeg_features)
    padding_masks = np.array(padding_masks)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')
        
    # Normalize data from 0 to 1 - training set
    scaler = MinMaxScaler()
    n_samples, n_epochs, n_features = eeg_features.shape
    eeg_features = eeg_features.reshape((n_samples * n_epochs, n_features))
    eeg_features = scaler.fit_transform(eeg_features)
    eeg_features = eeg_features.reshape((n_samples, n_epochs, n_features))
    
    # Get train eeg outcomes from patient outcomes
    recording_labels = []
    for patient_index in recording_patient_dictionary:
        recording_labels.append((outcomes[patient_index], cpcs[patient_index]))
    recording_labels = np.array(recording_labels)
    
    # Process tensors
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(eeg_features, dtype=torch.float32).to(device)
    y_train_out_tensor = torch.tensor(recording_labels[:,0], dtype=torch.int64).to(device)
    y_train_cpc_tensor = torch.tensor(recording_labels[:,1], dtype=torch.float32).to(device)
    pindex_train_tensor = torch.tensor(recording_patient_dictionary, dtype=torch.int64).to(device)
    mask_train_tensor = torch.tensor(padding_masks, dtype=torch.bool).to(device)
    batch_size = 16
    train_ds = TensorDataset(X_train_tensor, y_train_out_tensor, y_train_cpc_tensor, pindex_train_tensor, mask_train_tensor)
    input_size = X_train_tensor.shape[2]
    
    # Set random seeds
    random_seed = 15
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    # Train the models.
    model = Classifier(input_size=input_size, num_classes=2, n_embedding=64, num_heads=8, num_layers=3, dropout=0.4).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.99)
    train_out_model(model, train_dl, loss_fn, optimizer, scheduler, device, epochs=200)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.99)
    train_cpc_model(model, train_dl, loss_fn, optimizer, scheduler, device, epochs=200)
    
    # Save the models.
    save_challenge_model(model_folder, scaler, outcome_model.state_dict(), cpc_model.state_dict())

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    mne.set_log_level('WARNING')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):

    print('Processing patient ', patient_id)
    scaler = models['scaler']
    outcome_model_state_dict = models['outcome_model']
    cpc_model_state_dict = models['cpc_model']
    
    # Extract features.
    eeg_features, hours, masks = get_features(data_folder, patient_id)
    eeg_features = np.array(eeg_features)
    masks = np.array(masks)

    # Scale data to prepare as input
    n_samples, n_epochs, n_features = eeg_features.shape
    eeg_features = eeg_features.reshape((n_samples * n_epochs, n_features))
    eeg_features = scaler.transform(eeg_features)
    eeg_features = eeg_features.reshape((n_samples, n_epochs, n_features))
    
    # Set random seeds
    random_seed = 15
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    # Setup data for models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_test_tensor = torch.tensor(eeg_features, dtype=torch.float32).to(device)
    mask_test_tensor = torch.tensor(masks, dtype=torch.bool).to(device)
    batch_size = 16
    test_ds = TensorDataset(X_test_tensor, mask_test_tensor)
    input_size = X_test_tensor.shape[2]
    test_dl = DataLoader(test_ds, batch_size)
    
    # Load models
    outcome_model = Classifier(input_size=input_size, num_classes=2, n_embedding=64, num_heads=8, num_layers=3, dropout=0.4).to(device)
    outcome_model.load_state_dict(outcome_model_state_dict)
    cpc_model = CPC_Classifier(input_size=input_size, num_classes=2, n_embedding=64, num_heads=8, num_layers=3, dropout=0.4).to(device)
    cpc_model.load_state_dict(cpc_model_state_dict)

    # Evaluate with models
    pred_threshold = 0.62
    outcome, outcome_probability = evaluate_out_test(outcome_model, test_dl, device, pred_threshold)
    cpc = evaluate_cpc_test(cpc_model, test_dl, device)

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    resampling_frequency = 128
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)
    
    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data
        
    return data, resampling_frequency

# Extract features.
def get_features(data_folder, patient_id):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    num_recordings = len(recording_ids)
    all_eeg_features = []
    hours_list = []
    masks_list = []

    # Extract EEG features.
    eeg_channels = ['F4','C4','P4','O2','F3','C3','P3','O1','F8','T4','T6','F7','T3','T5','Fp1','Fp2','Fz','Pz','Cz']
    group = 'EEG'
    
    if num_recordings > 0:
        # limit to earliest recording for training and testing, we don't want data very far from point of cardiac arrest
        for i in range(num_recordings):
            recording_id = recording_ids[i]
            recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
            match = re.search(r'(\d+)_EEG$', recording_location)
            hour = int(match.group(1))

            if os.path.exists(recording_location + '.hea') and hour <= 72:
                data, channels, sampling_frequency = load_recording_data(recording_location)
                header = get_hea_string(recording_location)
                utility_frequency = get_utility_frequency_header(header)
                start_time = get_start_time_header(header)
                end_time = get_end_time_header(header)
                
                if all(channel in channels for channel in eeg_channels):
                    data, channel_names = reduce_channels(data, channels, eeg_channels)
                    data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                    # Create epochs
                    ch_types = ['eeg'] * len(channel_names)
                    info = mne.create_info(ch_names=channel_names, sfreq=sampling_frequency, ch_types=ch_types)
                    eeg = mne.io.RawArray(data, info)
                    epoch_duration = 5 * 60

                    if eeg.n_times >= epoch_duration * eeg.info['sfreq']:
                        events = mne.make_fixed_length_events(eeg, duration=epoch_duration)
                        epochs = mne.Epochs(eeg, events, event_id=None, tmin=0, tmax=epoch_duration, baseline=None, detrend=1)
                        epochs.drop_bad()
                        good_epoch_indices = epochs.selection

                        if len(good_epoch_indices) > 0:
                            if not hour in hours_list:
                                hours_list.append(hour)
                                eeg_features = list()
                                masks = list()
                                time_index = 5
                                while time_index < start_time:
                                    eeg_features.append(np.zeros(76))
                                    masks.append(True)
                                    time_index += 5
                                for i in range(len(epochs.events)):
                                    epoch = epochs[i].get_data()[0, :, :-1]
                                    eeg_features.append(get_eeg_features(epoch, 128).flatten())
                                    masks.append(False)
                                while end_time < 55 or len(eeg_features)<11:
                                    eeg_features.append(np.zeros(76))
                                    masks.append(True)
                                    end_time += 5
                                all_eeg_features.append(eeg_features)
                                masks_list.append(masks)
                            else:
                                #continue previous hour, so just concatenate all the samples and features, don't append hours anymore
                                #find the starting index on where to insert this recording since we continue from the previous hour
                                starting_index = start_time // 5
                                for i in range(len(epochs.events)):
                                    epoch = epochs[i].get_data()[0, :, :-1]
                                    all_eeg_features[-1][starting_index] = get_eeg_features(epoch, 128).flatten()
                                    masks_list[-1][starting_index] = False
                                    starting_index += 1

                                    
    else:
        all_eeg_features.append([[0] * 76 for _ in range(11)])
        masks_list.append([[False] for _ in range(11)])
        hours_list.append(1)
        all_eeg_features = np.array(all_eeg_features)
        masks_list = np.squeeze(np.array(masks_list), axis=-1)
        hours_list = np.array(hours_list)
    
    return all_eeg_features, hours_list, masks_list

# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm))

    return features

# Extract features from the EEG data.
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=0.1, fmax=4.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=4.0, fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=8.0, fmax=12.0, verbose=False)
        beta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean = np.nanmean(beta_psd, axis=1)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)

    features = np.array((delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean)).T

    return features

# Extract features from the ECG data.
def get_ecg_features(data):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        mean = np.mean(data, axis=1)
        std  = np.std(data, axis=1)
    elif num_samples == 1:
        mean = np.mean(data, axis=1)
        std  = float('nan') * np.ones(num_channels)
    else:
        mean = float('nan') * np.ones(num_channels)
        std = float('nan') * np.ones(num_channels)

    features = np.array((mean, std)).T

    return features

def get_hea_string(recording_location):
    with open(recording_location+'.hea', 'r') as f:
        header = [l.strip() for l in f.readlines() if l.strip()]
    return header

def get_utility_frequency_header(header):
    # Extract the utility frequency
    utility_frequency = None
    for item in header:
        if item.startswith('#Utility frequency:'):
            utility_frequency = int(item.split(': ')[1])
            break
    return utility_frequency

def get_start_time_header(header):
    # Extract the start time minutes
    start_time_minutes = None
    for item in header:
        if item.startswith('#Start time:'):
            start_time = item.split(': ')[1]
            start_time_minutes = int(start_time.split(':')[1])
            break
    return start_time_minutes

def get_end_time_header(header):
    # Extract the start time minutes
    end_time_minutes = None
    for item in header:
        if item.startswith('#End time:'):
            end_time = item.split(': ')[1]
            end_time_minutes = int(end_time.split(':')[1])
            break
    return end_time_minutes

def compute_tpr_at_fpr(labels, outputs, target_fpr=0.05):
    assert len(labels) == len(outputs)

    # Combine labels and outputs into a 2D array
    data = np.column_stack((labels, outputs))

    # Sort the data based on output scores in descending order
    sorted_data = data[data[:, 1].argsort()[::-1]]

    num_positives = np.sum(labels == 1)
    num_negatives = len(labels) - num_positives

    # Initialize variables
    tp = 0
    fp = 0
    tpr = 0.0

    for i in range(len(sorted_data)):
        if sorted_data[i, 0] == 1:
            tp += 1
        else:
            fp += 1

        current_fpr = fp / num_negatives
        current_tpr = tp / num_positives

        if current_fpr >= target_fpr:
            tpr = current_tpr
            break

    return tpr

def train_out_model(model, train_dl, loss_fn, optimizer, scheduler, device, epochs, num_folds=5, patience=10, best_auprc=0):
    print("Training model for outcome prediction, with cross validation...")
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=15)
    current_patience = 0
    
    for epoch in range(epochs):
        if current_patience < patience:
            print(f"Epoch {epoch + 1}/{epochs}...")
            y_pred_prob_list = list()
            y_true_list = list()
            patient_indices_list = list()

            train_loss_epoch = 0.0
            val_loss_epoch = 0.0
            
            for fold, (train_index, val_index) in enumerate(kf.split(train_dl.dataset)):
                train_dataset = torch.utils.data.Subset(train_dl.dataset, train_index)
                val_dataset = torch.utils.data.Subset(train_dl.dataset, val_index)

                train_dl_fold = torch.utils.data.DataLoader(train_dataset, batch_size=train_dl.batch_size, shuffle=True)
                val_dl_fold = torch.utils.data.DataLoader(val_dataset, batch_size=train_dl.batch_size)

                model.train()
       
                train_loss = 0
                for x, y, _, p_index, padding_mask in train_dl_fold:
                    optimizer.zero_grad()
                    outcomes = model.forward(x, padding_mask, device)
                    loss = loss_fn(outcomes, y.unsqueeze(1).float())
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                
                train_loss_epoch = train_loss_epoch + (train_loss / len(train_dl_fold))
                
                # Evaluate on the validation set
                val_loss, y_pred_prob, y_true, patient_indices = evaluate_out_val(model, val_dl_fold, loss_fn, device)
                val_loss_epoch += val_loss
                
                y_pred_prob_list.extend(y_pred_prob)
                y_true_list.extend(y_true)
                patient_indices_list.extend(patient_indices)
                
            scheduler.step()
            
            # Print progress and metrics
            print(f"Train Loss per epoch: {train_loss_epoch/num_folds:.4f} | Val Loss per epoch: {val_loss_epoch/num_folds:.4f}")

            # do majority voting here
            curr_index = 0
            patient_pred_prob = list()
            p_labels = list()
            
            while curr_index <= np.max(patient_indices_list):
                curr_pred_probs = list()
                curr_truths = list()
                for i in range(len(patient_indices_list)):
                    if patient_indices_list[i] == curr_index:
                        curr_pred_probs.append(y_pred_prob_list[i])
                        curr_truths.append(y_true_list[i])

                # compute average
                if len(curr_truths) > 0:
                    patient_pred_prob.append(np.mean(curr_pred_probs))
                    p_labels.append(curr_truths[0])
                curr_index+=1
            
            patient_pred_prob = np.array(patient_pred_prob)
            p_labels = np.array(p_labels)
            
            tpr_over_fpr = compute_tpr_at_fpr(p_labels, patient_pred_prob)
            auroc = roc_auc_score(p_labels, patient_pred_prob)
            auprc = average_precision_score(p_labels, patient_pred_prob)
            
            print("Patient-wise predictions over cross-validation: ")
            print("True Positive Rate over False Positive Rate:", tpr_over_fpr)
            print("AUROC:", auroc)
            print("AUPRC:", auprc)

            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            best_f1_score = 0
            best_accuracy = 0
            best_threshold = 0
            for threshold in thresholds:
                patient_pred = list()
                for pred_prob in patient_pred_prob:
                    if np.mean(pred_prob)>threshold:
                        patient_pred.append(1)
                    else:
                        patient_pred.append(0)
                        
                patient_pred = np.array(patient_pred)
                accuracy = accuracy_score(p_labels, patient_pred)
                f1 = f1_score(p_labels, patient_pred)
                if f1 > best_f1_score:
                    best_f1_score = f1
                    best_accuracy = accuracy
                    best_threshold = threshold
                
            print('Optimized threshold: ', best_threshold)
            print("Accuracy: ", best_accuracy)
            print("F1 Score: ", best_f1_score)
            
            # Check for early stopping
            if auprc > best_auprc:
                best_auprc = auprc
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= patience:
                    print(f"Early stopping at epoch {epoch+1}...")

def train_cpc_model(model, train_dl, loss_fn, optimizer, scheduler, device, epochs, num_folds=5, patience=10):
    print("Training model for outcome prediction, with cross validation...")
    
    best_val_loss = float('inf')
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=15)
    current_patience = 0
    
    for epoch in range(epochs):
        if current_patience < patience:
            print(f"Epoch {epoch + 1}/{epochs}...")

            y_pred_list = list()
            y_true_list = list()
            patient_indices_list = list()

            train_loss_epoch = 0.0
            val_loss_epoch = 0.0
            
            for fold, (train_index, val_index) in enumerate(kf.split(train_dl.dataset)):
                train_dataset = torch.utils.data.Subset(train_dl.dataset, train_index)
                val_dataset = torch.utils.data.Subset(train_dl.dataset, val_index)

                train_dl_fold = torch.utils.data.DataLoader(train_dataset, batch_size=train_dl.batch_size, shuffle=True)
                val_dl_fold = torch.utils.data.DataLoader(val_dataset, batch_size=train_dl.batch_size)

                model.train()
       
                train_loss = 0
                for x, _, y, p_index, padding_mask in train_dl_fold:
                    optimizer.zero_grad()
                    outcomes = model.forward(x, padding_mask, device)
                    loss = loss_fn(outcomes, y.unsqueeze(1).float())
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                
                train_loss_epoch = train_loss_epoch + (train_loss / len(train_dl_fold))
                
                # Evaluate on the validation set
                val_loss, y_pred_prob, y_true, patient_indices = evaluate_cpc_val(model, val_dl_fold, loss_fn, device)
                val_loss_epoch += val_loss
                
                y_pred_list.extend(y_pred_prob)
                y_true_list.extend(y_true)
                patient_indices_list.extend(patient_indices)
                
            scheduler.step()
            
            # Print progress and metrics
            print(f"Train Loss per epoch: {train_loss_epoch/num_folds:.4f} | Val Loss per epoch: {val_loss_epoch/num_folds:.4f}")
            
            # Check for early stopping
            if (val_loss_epoch/num_folds) < best_val_loss:
                best_val_loss = val_loss_epoch/num_folds
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= patience:
                    print(f"Early stopping at epoch {epoch+1}...")

def evaluate_out_val(model, val_dl, loss_fn, device):
    model.eval()
    val_loss = 0.0
    y_true = []
    y_pred_probs = []
    patient_indices = []

    with torch.no_grad():
        for x, y, _, p_index, key_padding_mask in val_dl:
            outcomes = model.forward(x, key_padding_mask, device)
            loss = loss_fn(outcomes, y.unsqueeze(1).float())
            val_loss += loss.item()
            y_pred_probs.extend(outcomes.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            patient_indices.extend(p_index.cpu())

    y_pred_probs = np.array(y_pred_probs)
    y_true = np.array(y_true)

    val_loss /= len(val_dl)
    patient_indices = np.array(patient_indices)

    return val_loss, y_pred_probs, y_true, patient_indices

def evaluate_cpc_val(model, val_dl, loss_fn, device):
    model.eval()
    val_loss = 0.0
    y_true = []
    y_preds = []
    patient_indices = []

    with torch.no_grad():
        for x, _, y, p_index, key_padding_mask in val_dl:
            outcomes = model.forward(x, key_padding_mask, device)
            loss = loss_fn(outcomes, y.unsqueeze(1).float())
            val_loss += loss.item()
            y_preds.extend(outcomes.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            patient_indices.extend(p_index.cpu())

    y_preds = np.array(y_preds)
    y_true = np.array(y_true)

    val_loss /= len(val_dl)
    patient_indices = np.array(patient_indices)

    return val_loss, y_preds, y_true, patient_indices

def evaluate_out_test(model, test_dl, device, pred_threshold=0.5):
    model.eval()
    y_pred_probs = []

    with torch.no_grad():
        for x, key_padding_mask in test_dl:
            outcomes = model.forward(x, key_padding_mask, device)
            y_pred_probs.extend(outcomes.cpu().numpy())

    y_pred_probs = np.squeeze(np.array(y_pred_probs))
    
    patient_pred_prob = np.mean(y_pred_probs)
    patient_pred = 0
    if patient_pred_prob > pred_threshold:
        patient_pred = 1
    else:
        patient_pred = 0
        
    return patient_pred, patient_pred_prob

def evaluate_cpc_test(model, test_dl, device):
    model.eval()
    y_preds = []

    with torch.no_grad():
        for x, key_padding_mask in test_dl:
            outcomes = model.forward(x, key_padding_mask, device)
            y_preds.extend(outcomes.cpu().numpy())

    y_preds = np.array(y_preds)
    patient_pred = np.mean(y_preds)
        
    return patient_pred

def compute_positional_encoding(max_len, d_model):
    position = np.arange(max_len)[:, np.newaxis]
    angle_rads = position / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return pos_encoding

class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_size, num_heads=8, n_embedding=128, dropout=0.3):
        super(TransformerEncoderBlock, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(n_embedding, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(n_embedding)
        self.feedforward_layer = nn.Sequential(
            nn.Linear(n_embedding, n_embedding*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_embedding*4, n_embedding)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding, key_padding_mask, device):
        positional_encoding = compute_positional_encoding(len(embedding[0]), len(embedding[0][0]))
        embedding = embedding + torch.from_numpy(positional_encoding.astype(np.float32)).to(device)

        residual = embedding
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.T
        embedding, attn_weights = self.multihead_attention(embedding, embedding, embedding, key_padding_mask=key_padding_mask)
        embedding += residual
        embedding = self.layer_norm(embedding)
        
        residual = embedding
        embedding = self.feedforward_layer(embedding)
        embedding = self.dropout(embedding)
        embedding += residual
        embedding = self.layer_norm(embedding)
        
        return embedding

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes=2, n_embedding=128, num_heads=8, num_layers=4, dropout=0.3):
        super(Classifier, self).__init__()
        
        self.embedding = nn.Linear(input_size, n_embedding)    
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderBlock(input_size, num_heads, n_embedding, dropout) 
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(n_embedding, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x, key_padding_mask, device):    
        embedding = self.embedding(x)
        for transformer_encoder in self.transformer_encoders:
            embedding = transformer_encoder(embedding, key_padding_mask, device)

        mean_embedding = []
        for recording_idx in range(len(key_padding_mask)):
            embedding_sum = torch.zeros(len(embedding[0][0]))
            count = 0
            for epoch_idx in range(len(key_padding_mask[recording_idx])):
                if key_padding_mask[recording_idx][epoch_idx] == False:
                    embedding_sum += embedding[recording_idx][epoch_idx].to('cpu')
                    count+=1
            mean_embedding.append(embedding_sum/count)
            
        mean_embedding = torch.stack(mean_embedding, dim=0).to(device)
        
        outcome = self.fc(mean_embedding)
        outcome = self.activation(outcome)
        return outcome

class CPC_Classifier(nn.Module):
    def __init__(self, input_size, num_classes=2, n_embedding=128, num_heads=8, num_layers=4, dropout=0.3):
        super(CPC_Classifier, self).__init__()
        
        self.embedding = nn.Linear(input_size, n_embedding)    
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderBlock(input_size, num_heads, n_embedding, dropout) 
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(n_embedding, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        #self.activation = nn.Sigmoid()

    def forward(self, x, key_padding_mask, device):    
        embedding = self.embedding(x)
        for transformer_encoder in self.transformer_encoders:
            embedding = transformer_encoder(embedding, key_padding_mask, device)

        mean_embedding = []
        for recording_idx in range(len(key_padding_mask)):
            embedding_sum = torch.zeros(len(embedding[0][0]))
            count = 0
            for epoch_idx in range(len(key_padding_mask[recording_idx])):
                if key_padding_mask[recording_idx][epoch_idx] == False:
                    embedding_sum += embedding[recording_idx][epoch_idx].to('cpu')
                    count+=1
            mean_embedding.append(embedding_sum/count)
            
        mean_embedding = torch.stack(mean_embedding, dim=0).to(device)
        
        outcome = self.fc(mean_embedding)
        #outcome = self.activation(outcome)
        return outcome
