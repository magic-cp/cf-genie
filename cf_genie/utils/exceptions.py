class ModelTrainingException(Exception):
    def __init__(self, model_name, *args, **kwargs) -> None:
        super().__init__('Model training failed: ' + model_name, *args, **kwargs)
        self.model_name = model_name
