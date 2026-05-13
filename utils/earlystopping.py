class EarlyStopping:
    def __init__(self, patience=25):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.best_params = None

    def step(self, val_loss, params):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_params = {k: v.copy() for k, v in params.items()}
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience