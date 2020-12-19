from pyod.models.auto_encoder import AutoEncoder

class Autocoder:
    ''' Model for autocoder '''
    def __init__(self, neurons: list):
        if len(neurons) == 0:
            exit(0)
        
        self.model = AutoEncoder(hidden_neurons = neurons)

    def train_model(self, X_train: object):
        self.model.fit(X_train)
    
    def predict(self, X_test: object) -> list:
        return self.model.decision_function(X_test)  # outlier scores
        
    def show_train_scores(self) -> list:
        return self.model.decision_scores_
