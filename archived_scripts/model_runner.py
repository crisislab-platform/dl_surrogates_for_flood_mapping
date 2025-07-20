# import logger
# from modules.lib.constants import RUN_DIR

# def predict_and_evaluate_only(run_id, model_name):
#     logger.info(f"Predicting and evaluating test data for run {run_id}")
#     run_dir = os.path.join(RUN_DIR, run_id)
#     model_file = os.path.join(run_dir, f"{model_name}.pt")
    
#     if not os.path.exists(model_file):
#         logger.error(f"Model file not found: {model_file}")
#         return
        
#     try:
#         # Load previously saved test data
#         x_test = np.load(f'{run_dir}/x_test.npy')
#         y_test = np.load(f'{run_dir}/y_test.npy')
        
#         # Run prediction and evaluation
#         predict_and_evaluate(run_id, run_dir, model_file, x_test, y_test)
#     except Exception as e:
#         logger.error(f"Error in prediction: {e}")
#         raise
    
# def predict_and_evaluate(run_id, run_dir, model_file_name, x_test, y_test):
#     logger.info(f"Predicting and evaluating test data for run {run_id}")
#     if not model_file_name or not os.path.exists(model_file_name):
#         logger.error(f"Model file not found: {model_file_name}")
#         return
        
#     logger.info(f"Loading model from {model_file_name}")
#     try:
#         # Set up device
#         device = configure_gpu()
        
#         # Load saved model
#         checkpoint = torch.load(model_file_name, map_location=device)
#         model_state_dict = checkpoint['model_state_dict']
        
#         # Recreate model instance
#         config_dict = checkpoint.get('config', {})
#         config = ModelConfig(**config_dict)
#         model = create_model(config, None)
        
#         if model is None:
#             logger.error("Failed to create model instance for loading")
#             return
            
#         # Prepare model for inference
#         model.load_state_dict(model_state_dict)
#         model.to(device)
#         model.eval()
        
#         # Log data shapes
#         logger.info(f"Test data shape: {x_test.shape}")
#         logger.info(f"Ground truth shape: {y_test.shape}")
        
#         # Save test data for future reference
#         np.save(f'{run_dir}/x_test.npy', x_test)
#         np.save(f'{run_dir}/y_test.npy', y_test)
        
#         # Convert data to torch tensors
#         x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
#         y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        
#         # Run inference in batches
#         predictions = []
#         start_time = time.time()
#         batch_size = 100
#         num_samples = len(x_test)
        
#         with torch.no_grad():
#             for i in range(0, num_samples, batch_size):
#                 batch_x = x_test_tensor[i:min(i + batch_size, num_samples)]
#                 pred = model(batch_x)
#                 predictions.append(pred.cpu().numpy())
                
#                 if (i + batch_size) % 1000 == 0:
#                     logger.info(f"Processed {i + batch_size}/{num_samples} predictions")
        
#         # Calculate time taken
#         end_time = time.time()
#         pred_time = end_time - start_time
        
#         # Process predictions
#         predictions = np.vstack(predictions)
#         y_test_np = y_test_tensor.cpu().numpy()
        
#         # Calculate metrics
#         mse = np.mean((y_test_np - predictions) ** 2)
#         mae = np.mean(np.abs(y_test_np - predictions))
#         rmse = np.sqrt(mse)
        
#         # Log results
#         logger.info(f"Prediction completed in {pred_time:.2f} seconds")
#         logger.info(f"Test MSE: {mse:.4f}")
#         logger.info(f"Test MAE: {mae:.4f}")
#         logger.info(f"Test RMSE: {rmse:.4f}")
        
#         # Update metrics tracking file
#         metrics_file = f'{RUN_DIR}/training_metrics.csv'
#         try:
#             df = pd.read_csv(metrics_file)
#             if run_id in df['run_id'].values:
#                 df.loc[df['run_id'] == run_id, 'pred_time'] = pred_time
#                 df.loc[df['run_id'] == run_id, 'pred_mse'] = mse
#                 df.loc[df['run_id'] == run_id, 'pred_mae'] = mae
#                 df.loc[df['run_id'] == run_id, 'pred_rmse'] = rmse
#                 df.to_csv(metrics_file, index=False)
#                 logger.info(f"Prediction results updated in {metrics_file}")
#             else:
#                 logger.warning(f"Run ID {run_id} not found in metrics file")
#         except Exception as e:
#             logger.error(f"Error updating metrics file: {e}")
            
#     except Exception as e:
#         logger.error(f"Error in prediction: {e}")
#         raise
    
#     finally:
#         # Clean up memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         gc.collect()
        
# def run_prediction(model, run_id, run_dir):
#     logger.info(f"Running prediction for model {model.config.model_name} with run ID {run_id}")
    
#     if not os.path.exists(run_dir):
#         logger.error(f"Run directory not found: {run_dir}")
#         return
        
#     try:
#         # Load test data
#         x_test = np.load(f'{run_dir}/x_test.npy')
#         y_test = np.load(f'{run_dir}/y_test.npy')
        
#         # Run prediction and evaluation
#         predict_and_evaluate(run_id, run_dir, model, x_test, y_test)
#     except Exception as e:
#         logger.error(f"Error in prediction: {e}")
#         raise
    