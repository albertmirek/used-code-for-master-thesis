import json
import os

import pickle
import sys

import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler



class SuperAiRecommenderHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        super(SuperAiRecommenderHandler, self).__init__()
        self.context = None
        self.initialized = False


    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        """
        #Initialize from context
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        src_path = os.path.join(model_dir, './')
        if src_path not in sys.path:
            sys.path.append(src_path)

        # Load configuration fileą
        from Config import Config
        config_path = model_dir + "/config.json"
        try:
            with open(config_path) as f:
                config_data = json.load(f)
                self.config = Config()
                self.config.__dict__.update(config_data)
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            raise

        #Load the model
        from Model import UserGruModel
        try:
            self.model = UserGruModel(self.config)
            self.model.load_state_dict(torch.load("model_state_dict.pth"))
            self.model.eval()
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

        #Load scalers and econders
        self.scalers = pickle.load(open(model_dir + "/scalers.pkl", "rb"))
        self.encoders = pickle.load(open(model_dir + "/encoders.pkl", "rb"))

        self.initialized = True


    def preprocess(self, data):
        if isinstance(data[0]['body'], str):
            input_data = json.loads(data[0]['body'])
        else:
            input_data = data[0]['body']

        sequences = input_data.get("data")
        processed_sequences = {}

        # Initialize dictionary to collect features
        for feature in self.config.model_columns:
            if feature == "session_id":
                continue
            processed_sequences[feature] = []

        for action in sequences:
            for feature, value in action.items():
                if feature in processed_sequences:  # Check if feature should be processed
                    if feature in self.encoders:
                        # Reshape value for encoding, handle None or missing values
                        value = -1 if value is None else value
                        transformed_value = self.encoders[feature].transform([[value]]).flatten()[0]
                        processed_sequences[feature].append(transformed_value)

                    if feature in self.scalers:
                        # Scale value, ensuring input is 2D
                        scaled_value = self.scalers[feature].transform([[value]]).flatten()[0]
                        processed_sequences[feature].append(scaled_value)

        # Convert lists to tensors and ensure all tensors are the same length for batching
        for feature in processed_sequences:
            processed_sequences[feature] = torch.tensor(processed_sequences[feature],
                                                        dtype=torch.float32 if feature in self.scalers else torch.long).unsqueeze(0)

        processed_sequences["lengths"] = torch.tensor([len(sequences)], dtype=torch.int64)
        return processed_sequences

    def postprocess(self, inference_output, top_k=5):
        probabilities = F.softmax(inference_output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)

        # Ensure top_indices is a flat list of indices
        top_indices = top_indices.squeeze()  # Adjust shape if necessary
        if top_indices.ndim > 1:
            top_indices = top_indices.view(-1)  # Flatten the indices if still more than 1D

        # Convert tensor to numpy and iterate to apply inverse_transform
        real_world_values = [self.encoders['product_id'].inverse_transform([idx])[0] for idx in top_indices.numpy()]

        result = [{
            "product_ids": real_world_values,
            "probabilities": top_probs.numpy().tolist()
        }]
        return result

    def inference(self, inputs):
        with torch.no_grad():
            if inputs is None:
                return None
            logits = self.model(**inputs)
        return logits

    def handle(self, data, context):
        """
        Handle request by preprocessing data, performing inference, and postprocessing the prediction output.
        :param data: Input data for prediction.
        :param context: Context containing model server system properties.
        :return: Prediction output.
        """
        try:
            self.context = context
            inputs = self.preprocess(data)  # Make sure preprocess returns both inputs and lengths
            if inputs is None:
                return json.dumps({"error": "No valid input data"})

            model_output = self.inference(inputs)  # Pass both inputs and lengths to inference
            output = self.postprocess(model_output)
            return output

        except Exception as e:
            print(f"Error handling the request: {str(e)}")
            return json.dumps({"error": str(e)})
