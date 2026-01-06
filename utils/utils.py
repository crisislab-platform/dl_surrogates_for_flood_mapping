import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Utils")

def check_if_already_run(args):
    from modules.lib.constants import RUN_DIR
    import os
    import pandas as pd
   
    if args.model == "LSTM_SRR_V1":
        rl_group = args.rl_id
    else:
        rl_group = args.rl_group
        
    logger.info(f"Logger info {rl_group}")
    
    if args.tuning_mode:
        metrics_file = f"{RUN_DIR}/{args.model}/tuning_metrics.csv"
        hyperparams = {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "input_time_len_h": args.input_time_len_h,
            "sampling_dist": args.sampling_dist,
            "n_clusters": args.n_clusters,
            "rl_group": rl_group,
            "lag": args.lag,
            "horizon": args.horizon,
            "patience": args.patience
        }
        compare_keys = ["batch_size", "learning_rate", "epochs", "patience", "lag", "horizon", "rl_group", "sampling_dist", "n_clusters", "input_time_len_h"]
    else:
        metrics_file = f"{RUN_DIR}/{args.model}/final_training_metrics.csv"
        # Define the hyperparameters to check for duplicates
        hyperparams = {
            # "batch_size": args.batch_size,
            # "learning_rate": args.learning_rate,
            # "epochs": args.epochs,
            # "input_time_len_h": args.input_time_len_h,
            # "sampling_dist": args.sampling_dist,
            # "n_clusters": args.n_clusters,
            "rl_group": rl_group,
            # "lag": args.lag,
            # "horizon": args.horizon,
            # "patience": args.patience,
            # "dropout": args.dropout,
            # "output_channel_size": args.output_channel_size,
            # "fc_layer_size": args.fc_layer_size
        }

        # Only keep keys that are relevant for comparison
        #compare_keys = ["batch_size", "learning_rate", "epochs", "patience", "lag", "horizon", "rl_group", "sampling_dist", "n_clusters", "input_time_len_h", "dropout", "output_channel_size", "fc_layer_size"]
        compare_keys = ["rl_group"]
    if not os.path.exists(metrics_file):
        return  # No previous runs, so continue

    try:
        df = pd.read_csv(metrics_file)
    except Exception as e:
        logger.warning(f"Could not read {metrics_file}: {e}")
        return
    
    # Prepare current run's hyperparameters as strings for comparison
    current_hyperparams = {k: hyperparams[k] for k in compare_keys if k in hyperparams}

    import ast
    for idx, row in df.iterrows():
        if row.get("model") != args.model:
            continue
        if args.tuning_mode and row.get("fold") != args.fold:
            continue
        try:
            row_hyperparams = ast.literal_eval(row.get("hyperparameters", "{}"))
        except Exception:
            continue
        # Only compare keys present in both
        if all(str(row_hyperparams.get(k)) == str(current_hyperparams.get(k)) for k in current_hyperparams):
            logger.warning("A run with the same hyperparameters already exists. Exiting to avoid duplicate runs.")
            exit(0)
