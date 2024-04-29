import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F



class UserGruModel(nn.Module):
    def __init__(self, config):
        super(UserGruModel, self).__init__()
        self.config = config
        self.hidden_units = config.hidden_units
        self.num_layers = config.num_layers

        # Entity embedding je pro items/users
        self.entity_embedding_dim = config.entity_embedding_dim
        self.context_embedding_dim = config.context_embedding_dim

        # Dynamic embedding toggle
        # Embeddings
        if "user_id" in self.config.model_columns:
            self.user_embedding = nn.Embedding(config.num_users, config.entity_embedding_dim)

        if "product_id" in self.config.model_columns:
            self.product_embedding = nn.Embedding(config.num_products, config.entity_embedding_dim)

        if "day_of_week" in self.config.model_columns:
            self.day_of_week_embedding = nn.Embedding(config.num_day_of_week, config.context_embedding_dim)  # Assuming 7 days

        if "week" in self.config.model_columns:
            self.week_embedding = nn.Embedding(53, config.context_embedding_dim)  # Assuming 53 weeks

        if "month" in self.config.model_columns:
            self.month_embedding = nn.Embedding(config.num_month, config.context_embedding_dim)  # Assuming 12 months

        if "brand_id" in self.config.model_columns:
            self.brand_embedding = nn.Embedding(config.num_brand_id + 1, config.context_embedding_dim)

        if "product_type_id" in self.config.model_columns:
            self.product_type_embedding = nn.Embedding(config.num_product_type_id + 1, config.context_embedding_dim)

        if "package_size" in self.config.model_columns:
            self.package_size_embedding = nn.Embedding(config.num_package_size + 1, config.context_embedding_dim)

        if "quality" in self.config.model_columns:
            self.quality_embedding = nn.Embedding(config.num_quality + 1, config.context_embedding_dim)


        self.rnn = nn.GRU(self.input_size(), self.hidden_units, self.num_layers, batch_first=True, dropout=config.dropout)

        self.fc = nn.Linear(self.hidden_units, config.num_products + 1)



    def input_size(self):
        """
        This method calculates the total size of the inputs to the RNN based
        on the chosen fusion method (right now statically just LINEAR)
        It is used to determine how many features each RNN cell should expect as a input
        The method is crucial for correctly setting up the RNN layer
        is called during model initialization to set the input size
        """
        #Calculate the size of the inputs to the RNN based on the combination type

        #Since using the dynamic feature toggle ...
        input_size = 0
        if "user_id" in self.config.model_columns:
            input_size += self.config.entity_embedding_dim

        if "product_id" in self.config.model_columns:
            input_size += self.config.entity_embedding_dim

        if "day_of_week" in self.config.model_columns:
            input_size += self.config.context_embedding_dim

        if "week" in self.config.model_columns:
            input_size += self.config.context_embedding_dim

        if "month" in self.config.model_columns:
            input_size += self.config.context_embedding_dim

        if "brand_id" in self.config.model_columns:
            input_size += self.config.context_embedding_dim

        if "product_type_id" in self.config.model_columns:
            input_size += self.config.context_embedding_dim

        if "package_size" in self.config.model_columns:
            input_size += self.config.context_embedding_dim

        if "quality" in self.config.model_columns:
            input_size += self.config.context_embedding_dim

        if self.config.num_numerical_feature is not None:
            input_size += self.config.num_numerical_feature

        return input_size

    def forward(self, **kwargs):
        """
        Defines how the model processes input data and generates output
        Called automatically by PyTorch during training or inference when
        data are passed to the model
        The computational logic is defined here
        """

        """
        These lines take the inidces for users, items, days of the week ,...
        and convert them to dense vector representations (Embeddings)
        output: tensor of shape [batch_size, sequence_length, embedding_dimension] 
        """

        embeddings = []

        if "user_id" in self.config.model_columns:
            user_embeddings = self.user_embedding(kwargs['user_id'])
            embeddings.append(user_embeddings)

        if "product_id" in self.config.model_columns:
            product_embeddings = self.product_embedding(kwargs['product_id'])
            embeddings.append(product_embeddings)

        if "day_of_week" in self.config.model_columns:
            day_of_week_embeddings = self.day_of_week_embedding(kwargs['day_of_week'])
            embeddings.append(day_of_week_embeddings)

        if "week" in self.config.model_columns:
            week_embeddings = self.week_embedding(kwargs['week'])
            embeddings.append(week_embeddings)

        if "month" in self.config.model_columns:
            month_embeddings = self.month_embedding(kwargs['month'])
            embeddings.append(month_embeddings)

        if "brand_id" in self.config.model_columns:
            brand_embeddings = self.brand_embedding(kwargs['brand_id'])
            embeddings.append(brand_embeddings)

        if "product_type_id" in self.config.model_columns:
            product_type_embeddings = self.product_type_embedding(kwargs['product_type_id'])
            embeddings.append(product_type_embeddings)

        if "package_size" in self.config.model_columns:
            package_size_embeddings = self.package_size_embedding(kwargs['package_size'])
            embeddings.append(package_size_embeddings)

        if "quality" in self.config.model_columns:
            quality_embeddings = self.quality_embedding(kwargs['quality'])
            embeddings.append(quality_embeddings)

        #Numerical features
        if "customer_price_cz" in self.config.model_columns:
            customer_price_cz_expanded = kwargs["customer_price_cz"].unsqueeze(-1)  # Adds an extra dimension
            embeddings.append(customer_price_cz_expanded)

        # Apply similar changes to other numerical features
        if "pocet_srdcovky" in self.config.model_columns:
            pocet_srdcovky_expanded = kwargs["pocet_srdcovky"].unsqueeze(-1)
            embeddings.append(pocet_srdcovky_expanded)

        if "rating_lifetime" in self.config.model_columns:
            rating_lifetime_expanded = kwargs["rating_lifetime"].unsqueeze(-1)
            embeddings.append(rating_lifetime_expanded)

        """
        Combines embeddings into a sungle input tensor that will be fed into the RNN
        depending on the configuration linear/linear-context etc.
        it concantenates embeddings along the last dimension (dim=-1)
        linear: combines only user and item
        output: [batch_size, sequence_length, total_embedding_size] total_em_size: sum of dimensions concantenated
        """

        rnn_input = torch.cat(embeddings, dim=-1)  # Ensure dimensions align as needed

        """
        packs a batch of variable-length sequences
        used to allow the RNN to only process the non-padded parts of batch
        output: PackedSequence: holds the data and batch sizes at each setp in the sequence
        packed_input is flattened form where sequences are concatenated end-toend and only include
        actual data points (no padding)
        """
        packed_input = pack_padded_sequence(rnn_input, kwargs['lengths'], batch_first=True, enforce_sorted=False)

        """
        Feeds the packed sequence input into the RNN
        processes and returns the packed sequence of outputs along with the final hidden states(ignored here)
        """
        packed_output, _ = self.rnn(packed_input)

        """
        Converts packed ouput back into a tensor of padded sequences, essentialy the reverse of pack_padded_sequence
        ouput: tensor shape [batch_size, max_sequence_lengths, hidden_units]
        hidden_units: dimension of RNN hidden layer (contains ouput of the RNN for each timestamp)
        """
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        """
        Identifies last non-padded element in each sequence (using lengths of the sequences) and extracts the ouputs
        Crucial for variable length sequences to ensure that the model uses the correct final hidden state for prediction
        ouput: tensor of shape [batch_size, hidden_units]
        """
        idx = (kwargs['lengths'] - 1).view(-1, 1).expand(len(kwargs['lengths']), output.size(2)).unsqueeze(1)
        decoded = output.gather(1, idx).squeeze(1)

        """
        Applies fully connected (linear) layer to the ouputs extracted from the last relevant timestamp
        output tensor [batch_size, num_items + 1]
        represents logits for each class (item) that the model predicts
        """
        logits = self.fc(decoded)
        return logits

    def compute_loss(self, logits, labels):
        """
        Computes the loss value
        will be called in training loop as:

        optimizer.zero_grad()  # Reset gradients
        logits = model.forward(inputs)  # Get model predictions
        loss = model.compute_loss(logits, labels)  # Calculate loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        loss = model.compute_loss(logits, label)
        """
        return F.cross_entropy(logits, labels)
