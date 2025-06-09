class DataManager:
    def __init__(self):
        self.train_idx = None
        self.validation_idx = None
        self.test_input = None
        self.test_output = None
        self.batch_size = None
        self.test_start_timestep = (17 * 4) - 2 * 4 # 17 hours, 4 timesteps per hour, -2 for the initialization period
        self.test_end_timestep = (65 * 4) - 2 * 4 # 65 hours, 4 timesteps per hour, -2 for the initialization period
        self.test_start_index = self.test_start_timestep -1 
        self.test_end_index = self.test_end_timestep - 1
        
    def get_batch(self, indices):
        pass
    