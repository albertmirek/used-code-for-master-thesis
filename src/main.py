import json
import os
import pickle
import subprocess
import time

import mlflow
import torch
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader

from modules.Dataset import SessionDataset, collate_fn
from modules.Model import UserGruModel
from modules.Config import Config


from utils.prepare_for_modeling import pre_process_dataset
from utils.train_test_split import train_test_split


def train_and_evaluate(model, train_loader, test_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()  # Set model to training mode
        total_train_loss = 0

        for data in train_loader:
            items, labels, lengths = data
            inputs = {k: v.to("cpu") for k, v in items.items()}  # Ensure data is on the correct device
            labels = labels.to("cpu")
            lengths = lengths.to("cpu")

            optimizer.zero_grad()
            logits = model(**inputs, lengths=lengths)
            loss = model.compute_loss(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_test_loss, accuracy, recall_at_k, mrr_at_k = evaluate(model, test_loader)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Log metrics to MLflow
        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
            "accuracy": accuracy,
            "recall_at_k": recall_at_k,
            "mrr_at_k": mrr_at_k,
            "elapsed_time_for_training": elapsed_time
        }, step=epoch)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, '
              f'Accuracy: {accuracy:.4f}, Recall@K: {recall_at_k:.4f}, MRR@K: {mrr_at_k:.4f}, '
              f'Elapsed Time: {elapsed_time:.2f}s')

def evaluate(model, data_loader, top_k=5):
    model.eval()
    total_loss = 0
    correct_topk = 0
    total_samples = 0
    sum_reciprocal_rank = 0

    with torch.no_grad():
        for data in data_loader:
            items, labels, lengths = data
            inputs = {k: v.to("cpu") for k, v in items.items()}
            labels = labels.to("cpu")
            lengths = lengths.to("cpu")

            logits = model(**inputs, lengths=lengths)
            probabilities = torch.softmax(logits, dim=1)
            loss = model.compute_loss(logits, labels)
            total_loss += loss.item()

            _, top_k_predictions = probabilities.topk(top_k, dim=1, largest=True, sorted=True)
            correct_topk += (top_k_predictions == labels.unsqueeze(1)).sum().item()
            total_samples += labels.size(0)

            # MRR@K calculation
            truth_positions = (top_k_predictions == labels.unsqueeze(1)).nonzero(as_tuple=True)[1]
            sum_reciprocal_rank += (1 / (truth_positions.float() + 1)).sum().item()

    avg_loss = total_loss / len(data_loader)
    topk_accuracy = correct_topk / total_samples
    recall_at_k = correct_topk / total_samples
    mrr_at_k = sum_reciprocal_rank / total_samples

    return avg_loss, topk_accuracy, recall_at_k, mrr_at_k


def main(config):
    # Read Data
    print("Loading data...")
    directory_path = f'./src/data/{config.dataset_version}/'
    df = pd.read_parquet(directory_path + 'processed_dataset.parquet')
    #df = pd.read_parquet(f"./src/data/0.1.8/test.parquet")

    print("Starting preprocess ...")
    df, encoders, scalers = pre_process_dataset(df, config)

    train_list, test_list = train_test_split(df, encoders)

    train_dataset = SessionDataset(train_list, config)
    test_dataset = SessionDataset(test_list, config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn)

    model = UserGruModel(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    mlflow.start_run()
    mlflow.log_params({
        "columns_used_for_training": config.model_columns,
        "entity_embedding_dim": config.entity_embedding_dim,
        "context_embedding_dim": config.context_embedding_dim,
        "learning_rate": config.learning_rate,
        "num_layers": config.num_layers,
        "hidden_units": config.hidden_units,
        "dropout": config.dropout,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
    })
    if RUN_ENVIRONMENT == "CI":
        mlflow.set_tag("gitlab_pipeline_id", os.getenv("CI_PIPELINE_ID", "Unknown pipeline ID"))
        mlflow.set_tag("gitlab_commit_sha", os.getenv("CI_COMMIT_SHA", "Unknown commit sha"))
        mlflow.set_tag("gitlab_project_url", os.getenv("CI_PROJECT_URL", "Unknown, project url"))
        mlflow.set_tag("gitlab_user_name", os.getenv("GITLAB_USER_NAME", "Unknown, user name"))

    print("Starting model training ...")
    train_and_evaluate(model, train_loader, test_loader, optimizer, config.num_epochs)

    #torch.save(model, "model.pth")
    torch.save(model.state_dict(), "model_state_dict.pth")

    #TODO adjust the save of the model, save only the state_dict
    # Log the model in MLflow
    mlflow.pytorch.log_model(model, "model")

    #Seem not to be working
    #mlflow.pytorch.log_model(model, "model", registered_model_name="UserGruModel")

    #Saving encoders into pickle file
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    #Saving scalers
    if scalers != {}:
        with open("scalers.pkl", "wb") as f:
            pickle.dump(scalers, f)


    with open('config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=4)

    # Use torch-model-archiver to package the model for TorchServe
    subprocess.run([
        "torch-model-archiver",
        "--model-name", "model",
        "--version", "1",
        "--serialized-file", "./model_state_dict.pth",
        "--model-file", "./src/modules/Model.py",
        "--handler", "./src/handler.py",
        "--extra-files", "encoders.pkl,scalers.pkl,config.json,./src/modules/Config.py",
        "--export-path", "./",
        "--force"
    ])

    # Log the .mar file as an artifact in MLflow
    mlflow.log_artifact("./model.mar")


RUN_ENVIRONMENT = os.getenv("RUN_ENVIRONMENT")
MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')

if __name__ == '__main__':
    config = Config()

    if RUN_ENVIRONMENT == "CI":
        # TODO - the mlflow server url was omited from the code
        mlflow.set_tracking_uri(f"")
    else:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri_docker)

    mlflow.set_experiment(config.experiment_name)
    main(config)

    #Cleanup
    if RUN_ENVIRONMENT == "CI":
        file_path = f"./src/data/{config.dataset_version}/processed_dataset.parquet"
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)  # Remove the file
            print(f"Deleted {file_path}")

