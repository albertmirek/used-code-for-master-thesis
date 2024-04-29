from torch.utils.data import Dataset
import torch


class SessionDataset(Dataset):
    def __init__(self, sessions, config):
        self.sessions = sessions
        self.sessions = [sess for sess in self.sessions if len(sess) > 0]
        self.config = config


    def __len__(self):
        """
        Provides the numver of items in the dataset
        required by PyTorch to determine the size of the dataset
        """
        return len(self.sessions)

    def __getitem__(self, index):
        """
        Retrieves single item from the dataset
        :param index: index of the item to retrieve
        :return: the tensors for userIds and productIds along with length of the session
        """
        #Fetcheds session at the specified index
        session = self.sessions[index]
        if len(session) == 0:
            raise ValueError(f"Session at index {index} is empty. This should be handled or cleaned.")

        items = {}

        if "user_id" in self.config.model_columns:
            items['user_id'] = torch.tensor(session['user_id'].values[:-1], dtype=torch.long)

        if "product_id" in self.config.model_columns:
            items['product_id'] = torch.tensor(session['product_id'].values[:-1], dtype=torch.long)

        if "day_of_week" in self.config.model_columns:
            items['day_of_week'] = torch.tensor(session['day_of_week'].values[:-1], dtype=torch.long)

        if "week" in self.config.model_columns:
            items['week'] = torch.tensor(session['week'].values[:-1], dtype=torch.long)

        if "month" in self.config.model_columns:
            items['month'] = torch.tensor(session['month'].values[:-1], dtype=torch.long)

        if "brand_id" in self.config.model_columns:
            items['brand_id'] = torch.tensor(session['brand_id'].values[:-1], dtype=torch.long)

        if "product_type_id" in self.config.model_columns:
            items['product_type_id'] = torch.tensor(session['product_type_id'].values[:-1], dtype=torch.long)

        if "package_size" in self.config.model_columns:
            items['package_size'] = torch.tensor(session['package_size'].values[:-1], dtype=torch.long)

        if "quality" in self.config.model_columns:
            items['quality'] = torch.tensor(session['quality'].values[:-1], dtype=torch.long)

        if "customer_price_cz" in self.config.model_columns:
            items['customer_price_cz'] = torch.tensor(session['customer_price_cz'].values[:-1], dtype=torch.float32)

        if "pocet_srdcovky" in self.config.model_columns:
            items['pocet_srdcovky'] = torch.tensor(session['pocet_srdcovky'].values[:-1], dtype=torch.float32)

        if "rating_lifetime" in self.config.model_columns:
            items['rating_lifetime'] = torch.tensor(session['rating_lifetime'].values[:-1], dtype=torch.float32)


        # Target label = product_id of the last event in the session
        label = torch.tensor(session['product_id'].values[-1], dtype=torch.long)

        return (items, label, len(session) - 1)


def collate_fn(batch):
    """
    Prepare the data batch by padding sequences to have the same length
    :param batch: list of tuples each containing userIds and productIds and session length
    as returned by __getitem__
    :return:
    """
    # Extract lengths for padding
    batched_items = {}
    labels = []
    lengths = []

    # Initialize lists in the dictionary for each feature
    for key in batch[0][0].keys():
        batched_items[key] = []

    # Collect data from each item in the batch
    for items, label, length in batch:
        for key in items:
            batched_items[key].append(items[key])
        labels.append(label)
        lengths.append(length)

    # Convert lists to tensors and handle padding if necessary
    for key in batched_items.keys():
        batched_items[key] = torch.nn.utils.rnn.pad_sequence(batched_items[key], batch_first=True, padding_value=0)

    # Stack labels and lengths into tensors
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return batched_items, labels, lengths

