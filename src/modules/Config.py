class Config:
    def __init__(self):
        # Environment Variables
        self.mlflow_tracking_uri_docker = "http://localhost:9005"
        self.experiment_name = "super-ai-recommender"

        """
        Dev parameters for modifying data feteches
        """
        self.do_fetch_data_locally = False
        self.skip_data_fetch = False


        """
        Pre-processing raw_data
        """
        self.max_length_to_truncate_to = 10
        self.max_session_length_upper_bound = 30
        self.max_session_duration_from_start_to_end_minutes = 30

        """
        Data & Dataset
        """
        # Columns to keep from dataset for modeling
        self.columns_to_keep = ["session_id", "user_id", "product_id", "brand_id", "product_type_id",
                                "customer_price_cz", "rating_lifetime"]
        # Columns that will be created from data ~ this array is used for generating additional embeddings
        self.columns_to_create = []
        # Final set of columns used for modeling
        self.model_columns = self.columns_to_keep + self.columns_to_create
        # Columns that will need to be scaled
        self.numerical_columns = ['customer_price_cz', 'pocet_srdcovky', 'rating_lifetime']
        #Columns that will need to be encoded
        self.categorical_columns = ["product_id", "user_id", "brand_id", "product_type_id", "package_size",
                                    "quality", "day_of_week", "month"]

        self.bucket_name = "recommender"
        self.dataset_version = "0.0.9"
        self.dataset_version_bucket_path = f"datasets/recommender-sequence-aware/{self.dataset_version}/"

        """
        Model Hyperparameters
        """
        # Embedding dimensions
        self.entity_embedding_dim = 50  # [50,100, 200, 300] Pro user/item
        self.context_embedding_dim = 5 #10

        # Layers parameters
        self.learning_rate = 0.001
        self.num_layers = 1 # If dropout this must be greater than 1
        self.hidden_units = 100
        self.dropout = 0  # Value [0.5] / None

        # Training Hyperparameters
        self.num_epochs = 1
        self.batch_size = 32

        #STATIC VARIABLES - do not touch; these are populated in code
        """
        STATIC VARIABLES
        do not touch; these are populated in code
        """

        self.num_products = None
        self.num_users = None
        self.num_day_of_week = 8
        self.num_month = 13
        self.num_week = None
        self.num_brand_id = None
        self.num_product_type_id = None
        self.num_package_size = None
        self.num_quality = None
        self.num_numerical_feature = None
        self.train_dataset_path = None
        self.test_dataset_path = None

    def to_dict(self):
        return self.__dict__





