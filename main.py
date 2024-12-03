from dataset import load_and_preprocess_data
from model import get_model
from train import perform_grid_search
from evaluate import evaluate_model

def main():
    X_train_seq, X_test_seq, y_train, y_test = load_and_preprocess_data()

    param_grid = {
        'units': [4, 8, 16, 32,64],
        'learning_rate': [0.0001, 0.001,0.009,0.01,0.02,0.05,0.1],
        'num_layers': [1, 2, 4],
        'activation': ['relu', 'tanh'],
        'batch_size': [1024, 2048,3096]
    }

    best_params = perform_grid_search(X_train_seq, y_train, param_grid, "grid_search", get_model)

    best_model = get_model(
        best_params['units'], best_params['learning_rate'],
        best_params['num_layers'], best_params['activation']
    )
    best_model.fit(X_train_seq, y_train, batch_size=best_params['batch_size'], epochs=10, verbose=0)

    evaluate_model(best_model, X_test_seq, y_test, "evaluation")

if __name__ == "__main__":
    main()

