def create_raster_dataset(batch_size=32, lag=8, horizon=1, epochs=10):
    #directory of the LISFLOOD-FP outputs i.e. Run2 is the outputs of Hydrograph A scenario, and so on...
    train_inun_files = []

    ##PROCESS TARGET DATA (Y_PARAM)
    train_inun_files += [each for each in os.listdir(lisflood_simulation_dir) if each.endswith('.wd') and not each.startswith(TEST_SUBSET) and not each.startswith(VALIDATION_SUBSET)]
    train_inun_files.sort()
    
    #list all the files that are not considered, i.e. LISFLOOD-FP initialisation time- about 2 hrs. Therefore, 8 correponding files need to be deleted (each file 15 min output)
    train_inun_files = [f for f.split('-') in train_inun_files if int(f.split('-')[1]) > 7]
        
    ###########sort target###############
    target = []
    for i in range(len(train_inun_files)):
        file_path = os.path.join(lisflood_simulation_dir, train_inun_files[i])
        data = rio.open(file_path)
        band = data.read(1)
        value = band.flatten()
        target.append(value)

    Y = np.array(target)
    Y[Y<0.3] = 0

    # Y should now be a 2104 x 581061 array, and target matrix for training process
    ## PROCESS TEST TARGET DATA
    test_inun_files = []
    test_inun_files += [each for each in os.listdir(lisflood_simulation_dir) if each.endswith('.wd') and each.startswith(TEST_SUBSET)]
    test_inun_files.sort()
    
    # similarly remove first 8 files
    test_excluded = ['Run1-0000.wd', 'Run1-0001.wd', 'Run1-0002.wd', 'Run1-0003.wd', 'Run1-0004.wd', 'Run1-0005.wd', 'Run1-0006.wd', 'Run1-0007.wd']
    for i in test_excluded:
        test_inun_files.remove(i)

    #Test target
    test_target = []
    for i in range(len(test_inun_files)):
        data = rio.open(lisflood_simulation_dir+test_inun_files[i])
        band = data.read(1)
        value = band.flatten()
        test_target.append(value)

    Y_test = np.array(test_target)
    Y_test[Y_test<0.3] = 0
    print(Y_test.shape)
    # Y_test should be a 266 x 581061 array for 2005 event
    
    
    #Validation target
    val_inun_files = [f for f in os.listdir(lisflood_simulation_dir) if f.endswith('.wd') and f.startswith(VALIDATION_SUBSET)]
    validation_excluded = ['Run9-0000.wd', 'Run9-0001.wd', 'Run9-0002.wd', 'Run9-0003.wd', 'Run9-0004.wd', 'Run9-0005.wd', 'Run9-0006.wd', 'Run9-0007.wd']
    for i in validation_excluded:
        val_inun_files.remove(i)
        
    val_target = []
    for i in range(len(val_inun_files)):
        data = rio.open(lisflood_simulation_dir+val_inun_files[i])
        band = data.read(1)
        value = band.flatten()
        val_target.append(value)
    Y_val = np.array(val_target)
    Y_val[Y_val<0.3] = 0
    print(Y_val.shape)
    # Y_val should be a 266 x 581061 array for 2005 event

    
    ######################### PREPARE X-PARAM DATA

    ####Import Precipitation/Discharge Data
    data_dir = bc_data_dir   #Directory of upstream flow data directory: Upstream_Flows_Run2.csv, Upstream_Flows_Run3.csv,
                                                                      #Upstream_Flows_Run4.csv,Upstream_Flows_Run5.csv,Upstream_Flows_Run6.csv,Upstream_Flows_Run7.csv,
                                                                      #Upstream_Flows_Run8.csv,Upstream_Flows_Run9.csv
                                                                      #These 8 files should be in this directory.
    data =[]
    data += [file for file in os.listdir(data_dir) if file.endswith('.csv') and not file.startswith(f'Upstream_Flows_{TEST_SUBSET}') and not file.startswith(f'Upstream_Flows_{VALIDATION_SUBSET}')] #Upstream_Flows_Run1.csv is the test data
    data.sort()
    print('Flow data files:',data)

    appended_data = []
    for f in data:
        train_df = pd.read_csv(data_dir+f)
        ##Shift the x parameter values back to represent antacedent hydrometeorological values, i.e. t-1, t-2, t-3 etc to t-8
        
        for i in range(1, lag+1):
            train_df['Upstream1-'+str(i)] = train_df['Upstream1'].shift(i)
            train_df['Upstream2-'+str(i)] = train_df['Upstream2'].shift(i)
            train_df['Upstream3-'+str(i)] = train_df['Upstream3'].shift(i)
        train_df = train_df[8:] # Remove the first 8 rows
        train_df = train_df.dropna()
        appended_data.append(train_df)

    appended_data = pd.concat(appended_data,ignore_index=True)
    # appended_data.to_csv('/home/cvssk/Carlisle_Resubmission/2005Event/Flows/Train/appended.csv')

    #Prepare test X_Param
    #Import Precipitation-Discharge Data
    test_df = pd.read_csv(f'{bc_data_dir}/Upstream_Flows_{TEST_SUBSET}.csv') #Upstream_Flows_Run1.csv for 2005 event
    
    ##Shift the x parameter values back to represent antacedent hydrometeorological values, i.e. t-1, t-2, t-3 etc
    for i in range(1, lag +1):
        test_df['Upstream1-'+str(i)] = test_df['Upstream1'].shift(i)
        test_df['Upstream2-'+str(i)] = test_df['Upstream2'].shift(i)
        test_df['Upstream3-'+str(i)] = test_df['Upstream3'].shift(i)
    test_df = test_df[8:] # Remove the first 8 rows
    test_df = test_df.dropna()
    
    all_data = pd.concat([appended_data, test_df],ignore_index=True)
    print('Length of the data after test:',len(all_data))
    
    
    # Prepare Validation X_Param
    validation_df = pd.read_csv(f'{bc_data_dir}/Upstream_Flows_{VALIDATION_SUBSET}.csv') #Upstream_Flows_Run9.csv for validation
    for i in range(1, lag +1):
        validation_df['Upstream1-'+str(i)] = validation_df['Upstream1'].shift(i)
        validation_df['Upstream2-'+str(i)] = validation_df['Upstream2'].shift(i)
        validation_df['Upstream3-'+str(i)] = validation_df['Upstream3'].shift(i)
    validation_df = validation_df[8:] # Remove the first 8 rows
    validation_df = validation_df.dropna()
    
    all_data = pd.concat([all_data, validation_df],ignore_index=True)
    print('Length of the data after validation:',len(all_data))
        
    scaler = StandardScaler() #MinMaxScaler(feature_range=(0, 1))
    all_data = scaler.fit_transform(all_data)
    
    test_count = len(test_df)
    val_count = len(validation_df)


    X_train = all_data
    X_val = X_train[-val_count:, :]
    X_train = X_train[:-val_count, :]
    X_test =  X_train[-test_count:, :]
    X_train = X_train[:-test_count, :]
    
    # Single time step prediction. So the shape of the input data should be (samples, time steps, features). 
    # Previous time steps are used as features with lagged features.
    x_train= X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    x_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    x_test= X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    del target
    del test_target
    del test_inun_files
    del train_inun_files
    del appended_data
    del test_df
    del all_data

    logger.info(f'X Train shape: {x_train.shape}, X Test shape: {x_test.shape}, Train target shape: {Y.shape}, Test target shape: {Y_test.shape}')
    logger.info('Data preprocessing complete!')

    # Convert to float32 for better performance
    x_train = x_train.astype('float32')
    Y = Y.astype('float32')
    x_test = x_test.astype('float32')
    Y_test = Y_test.astype('float32')
    x_val = x_val.astype('float32')
    Y_val = Y_val.astype('float32')

    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(Y, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

    # Create PyTorch datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f'PyTorch tensors created: X Train shape: {x_train_tensor.shape}, Y Train shape: {y_train_tensor.shape}')
    logger.info('PyTorch DataLoaders created successfully')

    return train_loader, val_loader, x_test_tensor, y_test_tensor, steps, features, outputs
