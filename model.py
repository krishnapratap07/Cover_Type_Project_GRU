from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Input
from tensorflow.keras.optimizers import Adam

def get_model(units, learning_rate, num_layers, activation):
    model = Sequential()
    model.add(Input(shape=(54, 1)))
    model.add(Bidirectional(GRU(units, return_sequences=(num_layers > 1))))

    for _ in range(num_layers - 1):
        model.add(Bidirectional(GRU(units, return_sequences=False if _ == num_layers - 2 else True)))

    model.add(Dense(units // 2, activation=activation))
    model.add(Dense(7, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
