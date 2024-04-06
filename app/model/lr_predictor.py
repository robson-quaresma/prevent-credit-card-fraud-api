import pickle

# Logistic Regression Predictor
class LRPredictor():

    def __init__(self) -> None:
        self.model = None
        
    def __load_model(self, model_path: str) -> None:
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def __make_prediction(self, data: object) -> int:
        prediction = self.model.predict(data) 
        return int(prediction)
    
    def predict(self, model_path: str, input_data: object) -> int:
        self.__load_model(model_path)
        return self.__make_prediction(input_data)
        