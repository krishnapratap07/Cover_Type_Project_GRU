import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from datasets import load_dataset

def load_and_preprocess_data():
    # Load dataset
    dataset = load_dataset("mstz/covertype", "covertype")["train"]
    data = dataset.to_pandas()

    # Separate features and target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # One-hot encode the target variable
    y = to_categorical(y)

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Expand dimensions for GRU input
    X_train_seq = np.expand_dims(X_train, axis=-1)
    X_test_seq = np.expand_dims(X_test, axis=-1)

    return X_train_seq, X_test_seq, y_train, y_test
