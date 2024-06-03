This repository contains our lab's implementation using the dataset from the George B. Moody PhysioNet Challenge 2023. It follows the code structure as was used in the challenge itself. Similar with the challenge, only the team_code.py has been modified.

Create folders in the same directory as the codes: mkdir training_data test_data model test_outputs
Installing dependencies: 'pip install -r requirements.txt'
Training the model: 'python train_model.py training_data model'
Running the model on validation and test sets: 'python run_model.py model test_data test_outputs'
Evaluating the model: 'python evaluate_model.py labels outputs scores.csv'
