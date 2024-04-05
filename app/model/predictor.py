import torch

class Predictor():

    def __init__(self):
        self.model = None
        self.prediction = None

    def load_model(self, model, model_path) -> None:
        # load ANN Model
        self.model = model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def make_prediction(self, data: torch.tensor) -> int:
        with torch.no_grad():
            self.prediction = self.model(data)

        probability = torch.sigmoid(self.prediction)
        return int((probability > 0.5).float())