
"""

    Many thanks to

    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

"""


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def criterion_met(self, validation_loss) -> bool:
        return validation_loss > (self.min_validation_loss + self.min_delta)

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif self.criterion_met(validation_loss):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
