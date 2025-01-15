# Create LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging
import time
from dataloader import FloodDataGenerator
import matplotlib.pyplot as plt
import json
import rasterio as rio
import numpy as np
import pandas as pd
from preprocessor import get_num_features
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LSTM_model_training")

# Constants
LAG = 8 # 8time steps of history. 2 hours  
FORECAST_HORIZON = 1 #Predict 1 time steps ahead. Next 15 minutes
TRAINING_BATCH_SIZE = 5000 # Number of samples per batch used for training

TRAIN_SUBSET_IDENTIFIER = 'train'
VAL_SUBSET_IDENTIFIER = 'validation'
PATIENCE = 2
EPOCHS = 10

def create_lstm_model(num_features):
    input_shape = (LAG, num_features)  # (timesteps, features)
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(FORECAST_HORIZON)  # Output shape matches horizon
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model
    
def create_data_generators(lag, horizon, batch_size):
    train_generator = FloodDataGenerator(batch_size,lag, horizon, TRAIN_SUBSET_IDENTIFIER)
    val_generator = FloodDataGenerator(batch_size, lag, horizon, VAL_SUBSET_IDENTIFIER)
    return train_generator, val_generator

# Train the model
def train_lstm_model(model, train_generator, val_generator):
    logger.info("Training LSTM model")
    model = create_lstm_model(num_features)
    start = time.time()
    
    call_backs = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=call_backs,
        verbose=1
    )

    end = time.time()
    logger.info(f"Training completed in {end-start} seconds")
    return model, history

def plot_model_training_history(history):
    # Plot the model history and save it as a figure and json file
    if not history:
        logger.error("Model history not found")
        return
    
    logger.info("Plotting model history")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('model_history.png')
    history_dict = history.history
    with open('model_history.json', 'w') as f:
        json.dump(history_dict, f)
        
# def predict(model,X_Test):
#   raster_reference = rio.open('/home/cvssk/Carlisle_Resubmission/2005Event/Target/Run2-0000.wd') #reference image for fixing raster dimensions
   
#   ## Make Predictions
#   for i in range(len(X_Test)):
#     x_test = X_Test[i]
#     x_test = x_test.reshape(1,1,X_Test.shape[1])
#     y_pred = model.predict(x_test)
    
#     y_pred.resize(raster_reference.height, raster_reference.width)
#     y_pred[y_pred<0.2] = 0

#     src = 
    
#     with rio.Env():
#         # Write an array as a raster band to a new 8-bit file. For
#         # the new file's profile, we start with the profile of the source
#         profile = src.profile

#         # And then change the band count to 1, set the
#         # dtype to uint8, and specify LZW compression.
#         profile.update(dtype=str(y_pred.dtype), count=1,compress='lzw')

#         with rio.open(TARGET_DIR +'CNN_2005_{:03}'".asc".format(i+8), 'w', **profile) as dst:
#         #with rio.open(tar_dir+fname+index+'.tif', 'w', **profile) as dst:
#             dst.write(y_pred, 1)

# def evaluate_model(model, test_generator):
#     logger.info("Evaluating the model")
#     predictions  = []
#     for batch_data in enumerate(test_generator):
#         logger.info("Processing batch {idx}")
#         x_test, y_test = batch_data
#         logger.info("Evaluating the model")
#         # Predict the next time step
#         predictions = model.predict(x_test)
#         # write the predictions to a csv file
#         out_df = pd.DataFrame(x_test.__index__(), columns = ['x','y','timestep','depth'])
#         predictions.to_csv('predictions.csv')
        
#         #write the prediction to a raster file
#     # Comapre the prediction with the actual value
#     predictions = np.reshape(predictions, -1)
#     # going to be a huge 
#     logger.info(f"Prediction: {prediction}")
#     logger.info(f"Actual: {y_test}")
#     logger.info("Model evaluation completed")
        
if __name__ == "__main__":
    logger.info("Model trainer started")
    num_features = get_num_features()
    train_generator, val_generator= create_data_generators(LAG, FORECAST_HORIZON, TRAINING_BATCH_SIZE)
    model = create_lstm_model(num_features)
    model, history = train_lstm_model(model, train_generator, val_generator)
    plot_model_training_history(history)
    model.save("lstm_model.h5")
    logger.info("Model saved as lstm_model.h5")
    logger.info("Model training completed")
    
    # Evaluate the model by making predictions on the test set
    # evaluate_model(model, val_generator)
    
    
    


