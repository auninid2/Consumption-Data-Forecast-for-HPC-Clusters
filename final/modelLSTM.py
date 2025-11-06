import TwoSidedAsymmetricHuberLoss as hl
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, BatchNormalization
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt


class modelLSTM:
    def __init__(self, modelName):
        self.__name = modelName

    """
    Prepare sequences for LSTM training.

    Parameters:
        data (pd.DataFrame): Historical data containing lagged features, temproal features, and rolling means.
        target_col (str): Column name to predict (e.g., 'Price').
        input_steps (int): Number of past hours to use as input sequence length.
        forecast_horizon (int): Number of future hours to predict.

    Returns:
        X (np.ndarray): Input sequences of shape (samples, input_steps, num_features).
        y (np.ndarray): Target sequences of shape (samples, forecast_horizon).
    """
    def __create_lstm_training_sequences(self, data, target, input_steps=48, forecast_horizon=1):
        X, y = [], []

        # Loop to create sequences within the historical data length
        max_start_idx = len(data) - input_steps - forecast_horizon + 1

        for i in range(max_start_idx):
            # Input: past `input_steps` hours of features
            X_seq = data.iloc[i : i + input_steps].values

            # Target: next `forecast_horizon` hours of price (known historical values)
            y_seq = target.iloc[i + input_steps : i + input_steps + forecast_horizon].values

            X.append(X_seq)
            y.append(y_seq)

        return np.array(X), np.array(y)

    """
    """
    def train(self, consumptionData, featureColumns, targetColumn, trainRatio=0.9):
        data = consumptionData.get()
        X = data[featureColumns]
        y = data[targetColumn]

        # Prepare sequences for LSTM training
        X_seq, y_targets = self.__create_lstm_training_sequences(X, y, input_steps=48, forecast_horizon=1)

        print("X_train shape:", X_seq.shape)  # (samples, 24, features)
        print("y_train shape:", y_targets.shape)  # (samples, 1)

        # Calculate the index to split the data
        # Ensure 'X_sequences' is defined from your actual data
        split_index = int(len(X_seq) * trainRatio)

        # Split the data chronologically
        X_train = X_seq[:split_index]
        y_train = y_targets[:split_index]

        X_test = X_seq[split_index:]
        y_test = y_targets[split_index:]

        print(f"Training set size (X_train): {X_train.shape}")
        print(f"Testing set size (X_test): {X_test.shape}")
        print(f"Training target size (y_train): {y_train.shape}")
        print(f"Testing target size (y_test): {y_test.shape}")

        X_scaler = MinMaxScaler()

        # Reshape X_train from 3D to 2D (samples * timesteps, features) for fitting
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1]) # Combines samples and timesteps dimensions
        X_scaler.fit(X_train_reshaped)

        # Transform both training and testing X data
        X_train_scaled = X_scaler.transform(X_train_reshaped).reshape(X_train.shape) # Transform and reshape back to 3D
        X_test_scaled = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # Scaler for y (target values)
        # y_train is already 2D (num_samples, forecast_horizon), so no complex reshaping needed
        y_scaler = MinMaxScaler()

        # Fit the y_scaler ONLY on the training target data
        y_scaler.fit(y_train)

        # Transform both training and testing y data
        y_train_scaled = y_scaler.transform(y_train)
        y_test_scaled = y_scaler.transform(y_test) # Transform y_test using scaler fitted on y_train

        print("\n--- After Scaling ---")
        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")
        print(f"y_train_scaled shape: {y_train_scaled.shape}")
        print(f"y_test_scaled shape: {y_test_scaled.shape}")

        delta = 0.01 # Huber transition point (tune around your normalized MAE)
        base_under_penalty_factor = 1.0 # Minimum multiplier for underpredictions
        peak_penalty_scalar = 5.0       # Higher values mean stronger penalty for missing HIGH peaks

        # For overpredicting low spikes/troughs:
        base_over_penalty_factor = 1.3  # Minimum multiplier for overpredictions
        trough_penalty_scalar = 4.5      # Higher values mean stronger penalty for overpredicting LOW troughs

        # Create an instance of custom loss function using the factory
        # This call returns the '_loss_fn' function itself.
        custom_loss_callable = hl.TwoSidedAsymmetricHuberLoss(
            delta=delta,
            base_under_penalty_factor=base_under_penalty_factor,
            peak_penalty_scalar=peak_penalty_scalar,
            base_over_penalty_factor=base_over_penalty_factor,
            trough_penalty_scalar=trough_penalty_scalar
        )

        model = Sequential()
        model.add(Conv1D(64, 2, activation='relu', padding='same', input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2, padding='valid'))
        model.add(Conv1D(128, 2, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2, padding='valid'))
        model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', dropout=0.1, recurrent_dropout=0.1))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))  # Output layer predicts 1 hour ahead

        model.compile(optimizer='adam', loss=custom_loss_callable, metrics=['mae', 'mse'])
        model.summary()

        early_stop = EarlyStopping(
            monitor='val_loss',   # watch validation loss
            patience=15,          # wait 15 epochs before stopping
            restore_best_weights=True  # restore model weights from the best epoch
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',  # Monitor validation loss
            factor=0.5,          # Reduce LR by 50%
            patience=10,         # Wait for 10 epochs without improvement
            min_lr=1e-7,         # Don't go below this LR
            verbose=1            # Print updates when LR changes
        )

        callbacks_list = [
            early_stop,
            lr_scheduler
        ]

        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=200,
            batch_size=32,
            validation_data=(X_test_scaled, y_test_scaled),
            callbacks=callbacks_list
        )

        model.save(self.__name + ".keras")
        joblib.dump(X_scaler, self.__name + "_minmax_X_scaler.gz")
        joblib.dump(y_scaler, self.__name + "_minmax_Y_scaler.gz")

        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss-over-epochs.png')

        predictions = model.predict(X_test_scaled)
        actual = y_scaler.inverse_transform(y_test_scaled)
        predictions = y_scaler.inverse_transform(predictions)

        plt.plot(actual, label='Actual')
        plt.plot(predictions, label='Forecasts')
        plt.legend()
        plt.title("Actual vs Forecasts")
        plt.savefig('actual-vs-forecasts.png')

    """
    Auxiliary function for forecasting
    """
    def __create_new_dates(self, date):
        future_dates = []
        for i in range(1, 2, 1):
            d = date + timedelta(hours=i)
            future_dates.append(d)
        return future_dates

    """
    Forecast horizon in hours
    """
    def forecast(self, consumptionData, targetColumn, horizon=24):
        model = load_model(self.__name + ".keras")
        X_scaler = joblib.load(self.__name + "_minmax_X_scaler.gz")
        y_scaler = joblib.load(self.__name + "_minmax_Y_scaler.gz")

        relevantCols = [targetColumn]
        colNum = 0 # columns except targetColumn
        laggedIdxList = []
        rollingMeanIdxList = []
        f = open(self.__name + '_training_features.txt', 'r')
        # display content of the file
        for line in f.read().splitlines():
            colNum += 1
            if targetColumn in line:
                if '_lagged_' in line:
                    laggedIdxList.append(int(line.removeprefix('price_lagged_')))
                if '_rollingMean_' in line:
                    rollingMeanIdxList.append(int(line.removeprefix('price_rollingMean_')))
            else:
                relevantCols.append(line)

        consumptionData.keepCols(pd.Index(relevantCols))

        for i in range(horizon):
            data = consumptionData.deepCopy()
            data.addLag([targetColumn], laggedIdxList)
            data.addRollingMean([targetColumn], rollingMeanIdxList)
            X = data.get()
            new_dates = self.__create_new_dates(X['_time'].iloc[-1]) # create 1 hour in the future
            X = X[168:] #moving 7 days forward because we need lagged features from previous week
            X = X.drop(['_time', targetColumn], axis=1)
            X = X.to_numpy().reshape(1, 48, colNum) #creating a 3d sequence of 24 hours
            X = X_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape) # normalization of data
            y = model.predict(X) # get forecasts
            y = y_scaler.inverse_transform(y) # getting real price values
            new_rows = pd.DataFrame(
                {'_time': new_dates, targetColumn: y.reshape(-1)}
            )
            consumptionData.appendRows(new_rows) # appending forecasted 1 hour and corresponding date to original dataframe
            consumptionData.dropHeadRows(1) # moving the original data ahead by 1 hour

        d = consumptionData.get()
        return d[targetColumn].tail(24) # the 24h forecast


if __name__ == '__main__':
    print("Error: Don't call modelLSTM.py directly, import it.")

