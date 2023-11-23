class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta

        self.counter = 0
        self.best_loss = None
        self.stop = False
        self.checkpoint_model = None

    def __call__(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.checkpoint(model)
        elif loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience: self.stop = True
        else:
            self.best_loss = loss
            self.checkpoint(model)
            self.counter = 0
        
        return self.stop
    
    def checkpoint(self, model):
        self.checkpoint_model = model
