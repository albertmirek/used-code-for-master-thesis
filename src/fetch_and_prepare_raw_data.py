import concurrent.futures
import os.path
from google.cloud import storage
from google.oauth2 import service_account
import glob
import os
import pandas as pd

from modules.Config import Config
from utils.pre_process_raw_data import pre_process_raw_data


def download_blob(blob, config):
    if blob.name.endswith('/'):
        print("Skipped a directory entry")
        return  # Skip directory entries, which end with '/'
    local_path = f'./src/data/{config.dataset_version}/raw_data/{blob.name.split("/")[-1]}'
    try:
        blob.download_to_filename(local_path)
        print(f'Downloaded {blob.name} to {local_path}')
    except Exception as e:
        print(f'Error downloading {blob.name}: {e}')

def fetch_data(config):
    directory_path = f"./src/data/{config.dataset_version}/raw_data/"

    # Check if the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' was created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    # SetUp credentials
    service_account_path = "./src/service_key.json"
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path,
    )

    # Instantiating a client using your credentials
    client = storage.Client(credentials=credentials, project=credentials.project_id)

    # Download dataset files
    bucket_name = config.bucket_name
    prefix = config.dataset_version_bucket_path

    bucket = client.get_bucket(bucket_name)

    # List all objects that start with the prefix
    blobs = bucket.list_blobs(prefix=prefix)

    #Wrapper for download function to insert config
    def download_blob_wrapper(blob):
        download_blob(blob, config)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        print("executing downloads")
        # Submit tasks to the executor for each blob
        executor.map(download_blob_wrapper, blobs)


def main(config):

    if RUN_ENVIRONMENT == "CI" and config.skip_data_fetch is False:
        print("Fetching raw data...")
        fetch_data(config)
    elif config.do_fetch_data_locally is True:
        print("Fetching raw data...")
        fetch_data(config)
    else:
        print("Data is not fetching, change config.do_fetch_data_locally to fetch it")

    parquet_files = glob.glob(f'./src/data/{config.dataset_version}/raw_data/*')
    dataframes = [pd.read_parquet(pf) for pf in parquet_files]

    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(dataframes, ignore_index=True)
    print("Downloaded raw dataset shape", df.shape)
    print("Starting to preprocess raw data for recommender")
    df = pre_process_raw_data(df, config)
    print("Final dataset shape after all pre-processment on raw data", df.shape)

    # Delete fetched dataset to save space
    if RUN_ENVIRONMENT == "CI":

        directory_path = f"./src/data/{config.dataset_version}/raw_data/"
        # Iterate over each entry in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)  # Create full file path
            print("to delete:", file_path)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # Remove the file
                    print(f"Deleted {file_path}")
                else:
                    print(f"Skipped {file_path}, as it is not a file")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    df.to_parquet(f"./src/data/{config.dataset_version}/processed_dataset.parquet")
    print("Data for modelling saved!")


RUN_ENVIRONMENT = os.getenv("RUN_ENVIRONMENT")
if __name__ == '__main__':
    config = Config()
    print("Dataset version used:", config.dataset_version)
    print("Preparing dataset from raw data")
    main(config)
