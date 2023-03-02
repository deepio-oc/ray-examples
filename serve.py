from ray import serve
import numpy as np
from typing import Dict
from starlette.requests import Request
import tensorflow as tf


TRAINED_MODEL_PATH = "./mnist_model.h5"

@serve.deployment
class TFMnistModel:
    def __init__(self, model_path: str):
        import tensorflow as tf

        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)

    async def __call__(self, starlette_request: Request) -> Dict:
        # Step 1: transform HTTP request -> tensorflow input
        # Here we define the request schema to be a json array.
        input_array = np.array((await starlette_request.json())["array"])
        reshaped_array = input_array.reshape((1, 28, 28))

        # Step 2: tensorflow input -> tensorflow output
        prediction = self.model(reshaped_array)

        # Step 3: tensorflow output -> web output
        return {"prediction": np.argmax(prediction.numpy().tolist()), "file": self.model_path}

mnist_model = TFMnistModel.bind(TRAINED_MODEL_PATH)

