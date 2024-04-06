import torch

class PCCFPredictor():

    def __init__(self):
        self.model = None
        self.prediction = None

    def __load_model(self, model: object, model_path: str) -> None:
        self.model = model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def __make_prediction(self, data: torch.tensor) -> int:
        with torch.no_grad():
            self.prediction = self.model(data)

        probability = torch.sigmoid(self.prediction)
        return int((probability > 0.5).float())
    
    def predict(self, model: object, model_path: str, input_data: torch.tensor) -> int:
        self.__load_model(model, model_path)
        return self.__make_prediction(input_data)