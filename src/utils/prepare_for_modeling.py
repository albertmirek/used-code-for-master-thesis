import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler, OrdinalEncoder

import pandas as pd


def pre_process_dataset(df, config):
    df.groupby(["session_id", "created_at"])
    df['created_at'] = pd.to_datetime(df["created_at"])

    print("Extraction of new features if specified by config")
    # Extract day_of_week and month
    if "day_of_week" in config.columns_to_create:
        df['day_of_week'] = df['created_at'].dt.dayofweek  # Monday=0, Sunday=6
        #config.num_day_of_week = len(df['day_of_week'].unique())

    if "week" in config.columns_to_create:
        df["week"] = df["created_at"].dt.week
        config.num_week = len(df["week"].unique())

    if "month" in config.columns_to_create:
        df['month'] = df['created_at'].dt.month
        #config.num_month = len(df['month'].unique())
        print("Num month", config.num_month)


    columns_to_keep = config.columns_to_keep + config.columns_to_create
    df = df[columns_to_keep]


    #categorical_features_to_encode = ["product_id", "user_id"]
    #This should take columns categorical columns to encode
    mask = np.in1d(np.array(config.columns_to_keep), np.array(config.categorical_columns))
    categorical_features_to_encode = np.array(config.columns_to_keep)[mask]
    encoders = {}

    for feature in categorical_features_to_encode:
        if feature == "product_id":
            encoder = LabelEncoder()
            df[feature] = encoder.fit_transform(df[feature])
            encoders[feature] = encoder
            print("Encoded categorical feature: ", feature)
        else:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df[feature] = encoder.fit_transform(df[[feature]])  # Double square brackets to make it a DataFrame
            encoders[feature] = encoder
            print("Encoded categorical feature: ", feature)

    config.num_users = len(df['user_id'].unique())
    config.num_products = len(df['product_id'].unique())
    config.max_session_length = 40  # TODO #Right now not used

    if "brand_id" in config.model_columns and df["brand_id"].unique().size:
        config.num_brand_id = len(df["brand_id"].unique())

    if "product_type_id" in config.model_columns and df["product_type_id"].unique().size:
        config.num_product_type_id = len(df["product_type_id"].unique())

    if "package_size" in config.model_columns and df["package_size"].unique().size:
        config.num_package_size = len(df["package_size"].unique())

    if "quality" in config.model_columns and df["quality"].unique().size:
        config.num_quality = len(df["quality"].unique())

    # Starting scaling
    numerical_mask = np.in1d(np.array(config.columns_to_keep), np.array(config.numerical_columns))
    numerical_feature_to_scale = np.array(config.columns_to_keep)[numerical_mask]
    scalers = {}

    for num_feature in numerical_feature_to_scale:
        if num_feature == "customer_price_cz":
            scaler = RobustScaler()
            df[num_feature] = scaler.fit_transform(df[[num_feature]])
            scalers[num_feature] = scaler
            print("Scaled numerical feature: ", num_feature, "with Robust scaler")
        else:
            scaler = StandardScaler()
            df[num_feature] = scaler.fit_transform(df[[num_feature]])
            scalers[num_feature] = scaler
            print("Scaled numerical feature: ", num_feature, "with Standard scaler")


    if len(numerical_feature_to_scale) > 0:
        config.num_numerical_feature = len(numerical_feature_to_scale)

    return df, encoders, scalers
