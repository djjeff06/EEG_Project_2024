## Licensing

This project contains code licensed under the BSD 2-Clause License and additional modifications by iDSSP Lab which are licensed under BSD 2-Clause License.

### Original Code License

The original code is licensed under the BSD 2-Clause License. See LICENSE for details.

### Modifications License

Modifications and additional code introduced by iDSSP Lab are licensed under BSD 2-Clause License. See LICENSE_NEW for details.

----------------------------------------------------------------------------------------------

This repository contains our lab's implementation using the dataset from the George B. Moody PhysioNet Challenge 2023. It follows the code structure as was used in the challenge itself. Similar with the challenge, only the team_code.py has been modified. Using an internal data split on the publicly available training set, with 80% training and 20% testing, we trained our model using cross validation and was able to achieve 0.82 AUROC, 0.90 AUPRC, 0.73 Accuracy, 0.79 F1-measure when evaluated with the internal 20% test set from the public training set. The rest of our processing steps are discussed in more detail in the draft article.

Create folders in the same directory as the codes: mkdir training_data test_data model test_outputs  
Installing dependencies: 'pip install -r requirements.txt'  
Training the model: 'python train_model.py training_data model'  
Running the model on validation and test sets: 'python run_model.py model test_data test_outputs'  
Evaluating the model: 'python evaluate_model.py labels outputs scores.csv'
