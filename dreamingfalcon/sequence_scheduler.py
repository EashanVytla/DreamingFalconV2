class AdaptiveSeqLengthScheduler:
    def __init__(self, initial_length, max_length, patience=3, threshold=0.01):
        self.current_length = initial_length
        self.max_length = max_length
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.plateau_counter = 0
        
    def step(self, current_loss):
        if current_loss < self.best_loss * (1 - self.threshold):
            self.best_loss = current_loss
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
            
        if self.plateau_counter >= self.patience:
            self.current_length = min(2 * self.current_length, self.max_length)
            self.plateau_counter = 0
            self.best_loss = float('inf')
    
        return self.current_length
    
    def step(self, current_loss, optimizer):
        if current_loss < self.best_loss * (1 - self.threshold):
            self.best_loss = current_loss
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
            
        if self.plateau_counter >= self.patience:
            optimizer.param_groups[0]['lr'] *= 0.25
            self.current_length = min(2 * self.current_length, self.max_length)
            self.plateau_counter = 0
            self.best_loss = float('inf')
    
        return self.current_length