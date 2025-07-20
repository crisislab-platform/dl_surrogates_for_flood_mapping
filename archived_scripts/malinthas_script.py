# %%
import rasterio as rio
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler 

# %%
def load_data():
    print('Loading LISFLOOD data')

    # Preprocess water depth data
    lisflood_simulation_dir = "/home/91/23016891/projects/Rapid_FloodModelling_CNN/Data/DEM5m_2D"

    # Need to remove first 8 files from each run as they are not useful. A run is a simulation of a flood event. 
    # File start with Run2-0000.wd, Run2-0001.wd, Run2-0002.wd, Run2-0003.wd, Run2-0004.wd, Run2-0005.wd, Run2-0006.wd, Run2-0007.wd. There is Run2 to Run9 in the simulation output.
    inundation_files = {}
    for i in range(1, 10):
        print(f'Processing Run{i}')
        inun_files = [file for file in os.listdir(lisflood_simulation_dir) if file.endswith('.wd') and file.startswith(f"Run{i}")]
        inundation_files[f"Run{i}"] = inun_files if len(inun_files) > 0 else []
    
    target = []
    test_target = []
    # Iterate the inundation_files dictionary
    for key, inun_files in inundation_files.items():
        inun_files.sort()
        print(f'Processing {key} with {len(inun_files)} files')
        inun_files = inun_files[8:] # remove first 8 files
        for i in range(len(inun_files)):
            data = rio.open(os.path.join(lisflood_simulation_dir, inun_files[i]))
            band = data.read(1)
            value = band.flatten()
            if key == 'Run1':
                test_target.append(value)
                continue
            target.append(value)

    Y_train = np.array(target)
    Y_test = np.array(test_target)
    Y_train[Y_train<0.3] = 0
    Y_test[Y_test<0.3] = 0

    # Preprocess input features
    # Import Precipitation/Discharge Data
    file_dir = "/home/91/23016891/projects/Rapid_FloodModelling_CNN/Data/"
    csv_files = [file for file in os.listdir(file_dir) if file.endswith('.csv')]

    # Iterate the csv_files
    csv_files.sort()
    df_train = None   
    for i in range(len(csv_files)):
        if i == 0:
            continue
        df = pd.read_csv(os.path.join(file_dir, csv_files[i]))
        df = preprocess_data(df)
        if df_train is not None:
            df_train = pd.concat([df_train, df], ignore_index=True)
        else:
            df_train = df

    # Add test data at the end
    df_test = pd.read_csv(os.path.join(file_dir, csv_files[0]))
    df_test = preprocess_data(df_test)
    df_train = pd.concat([df_train, df_test], ignore_index=True)
    print('Data loading complete')
    return df_train, Y_train, Y_test

def preprocess_data(df_train):
    print('Preprocessing data')
    scaler = MinMaxScaler()
    # Scale the data
    scaler_columns = ["Upstream1", "Upstream2", "Upstream3"]
    df_scaled = df_train.copy()
    df_scaled[scaler_columns] = scaler.fit_transform(df_train[scaler_columns])
    for i in range(1, 9):
        for col in [c for c in df_train.columns if c != 'Time']:
            df_scaled[f'lag_{i}_{col}'] = df_scaled[col].shift(i)
    df_scaled = df_scaled.dropna()
    print('Data preprocessing complete')
    return df_scaled

def split_data(df_train):
    print('Splitting data into training and testing sets')
    # Split the data into training and testing
    # Upstream_Flows_Run1 is used as the test data which has 274 time steps. So need to subtract 274 from the length of training data
    X_train = df_train.iloc[0:2104, :]
    X_test = df_train.iloc[2104:, :]

    X_train = X_train.drop(columns=['Time'])
    X_test = X_test.drop(columns=['Time'])
    # Reshape the data
    X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
    print('Data splitting complete')
    return X_train, X_test

# %%
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import timeit

from matplotlib import pyplot as plt

def CNN_Model(x_train, Y, x_test, Y_test, steps, features, outputs):
    '''
    Two layered conv network
    '''
    print('Running the CNN model...')
    model = Sequential()
    model.add(Conv1D(32, kernel_size=1, activation='relu', input_shape=(steps, features)))
    model.add(Conv1D(128, activation='relu', kernel_size=1))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(outputs))
    optimizer = Adam(learning_rate=0.01)
    model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)
    print(model.summary())
    # plot_model(model, to_file='/home/91/23016891/projects/Rapid_FloodModelling_CNN/CNN_Graph.png', dpi=1200)
    # Start time
    start = timeit.default_timer()
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    history = model.fit(x_train, Y, validation_data=(x_test, Y_test), batch_size=10, callbacks=[monitor], verbose=0, epochs=100)
    # Stop time
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    # Plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    print('CNN model training complete')
    return model


# CNN model with Batch normalization and dropout layers...Used for 2015 event modelling
def CNN_Model_BN(x_train, Y, x_test, Y_test, steps, features, outputs):
    '''
    Two layered conv network
    '''
    print('Running the CNN model with Batch Normalization...')
    model = Sequential()
    model.add(Conv1D(32, kernel_size=1, input_shape=(steps, features)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(128, kernel_size=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(32, kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(outputs))
    optimizer = Adam(lr=0.01)
    model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)
    print(model.summary())
    plot_model(model, to_file='/home/91/23016891/projects/Rapid_FloodModelling_CNN/CNN_BN_Graph.png')
    # Start time
    start = timeit.default_timer()
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    history = model.fit(x_train, Y, validation_data=(x_test, Y_test), batch_size=32, callbacks=[monitor], verbose=0, epochs=100)
    # Stop time
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    # Plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    print('CNN model with Batch Normalization training complete')
    return model


# %%
print('Loading data...')
df_train, Y_train, Y_test = load_data()
print('Data loaded and preprocessed')
print('Splitting data...')
X_train, X_test = split_data(df_train)
print('Data split complete')
Y_train = Y_train[:, :10]
Y_test = Y_test[:, :10]

steps = X_train.shape[1]
features = X_train.shape[2]
outputs = Y_train.shape[1]

print(f'X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}, X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}')

#Y train's shape is (266, 5000000). I want to reduce this to 10 so that I can locally train this

model = CNN_Model(X_train, Y_train, X_test, Y_test, steps, features, outputs)

# %%
