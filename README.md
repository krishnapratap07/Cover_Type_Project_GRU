# Covertype Classification Using GRU

This project provides a Deep Learning algorithm for classifying forest cover types using the Covertype dataset. The model is a **Bidirectional GRU**, and the algorithm includes data preprocessing, hyperparameter tuning, model training, and evaluation.

## Project Structure

The repository includes the following Python scripts.

```
├── dataset.py        # Data loading and preprocessing
├── model.py          # GRU model creation
├── train.py          # Hyperparameter tuning and training logic
├── evaluate.py       # Model evaluation and report generation
├── main.py           # Entry point for the program
└── README.md         # Instructions to run the project
```
## Prerequisites
Python 3.7 or higher
## Installation
1. Clone the repository:
    ```
   git clone https://github.com/krishnapratap07/Cover_Type_Project_GRU.git
   cd Cover_Type_Project_GRU
    ```
2. Install the required python package
    ```
   pip install -r requirements.txt
   ```
## Usuage
### Run the Main Script
Once the installation is complete, you can run the main 
program that will train the model and evaluate the results.
To execute the script, run the following command in your terminal.
```
python main.py
```