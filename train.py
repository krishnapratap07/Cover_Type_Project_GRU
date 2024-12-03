import os
import json
from itertools import product
import random
import pandas as pd

def perform_grid_search(X_train_seq, y_train, param_grid, output_dir, get_model_func):
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/grid_search_progress.json"

    # Initialize or load results
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            grid_search_results = json.load(f)
    else:
        grid_search_results = []

    completed_combinations = {
        (res['units'], res['learning_rate'], res['num_layers'], res['activation'], res['batch_size'])
        for res in grid_search_results
    }

    param_combinations = list(product(
        param_grid['units'],
        param_grid['learning_rate'],
        param_grid['num_layers'],
        param_grid['activation'],
        param_grid['batch_size']
    ))
    random_combinations = random.sample(param_combinations, 30)

    for (units, lr, num_layers, activation, batch_size) in random_combinations:
        if (units, lr, num_layers, activation, batch_size) in completed_combinations:
            continue

        print(f"Testing: units={units}, lr={lr}, layers={num_layers}, activation={activation}, batch_size={batch_size}")
        model = get_model_func(units, lr, num_layers, activation)

        history = model.fit(
            X_train_seq, y_train,
            validation_split=0.2, batch_size=batch_size, epochs=10, verbose=0
        )

        val_accuracy = max(history.history['val_accuracy'])
        grid_search_results.append({
            'units': units, 'learning_rate': lr, 'num_layers': num_layers,
            'activation': activation, 'batch_size': batch_size, 'val_accuracy': val_accuracy
        })

        with open(results_file, "w") as f:
            json.dump(grid_search_results, f, indent=4)

    results_df = pd.DataFrame(grid_search_results)
    results_df.to_csv(f"{output_dir}/exp_results.csv", index=False)

    best_params = results_df.loc[results_df["val_accuracy"].idxmax()].to_dict()
    return best_params
