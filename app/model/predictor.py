import torch

class Predictor():

    def __init__(self):
        self.model = None
        self.prediction = None

    def load_model(self, model):
        # load ANN Model
        self.model = model()
        self.model.load_state_dict(torch.load('pccf_model.pth'))
        self.model.eval()
    
    def make_prediction(self, data):
        with torch.no_grad():
            self.prediction = self.model(data)

        probability = torch.sigmoid(self.prediction)
        return int((probability > 0.5).float())