import sys

import numpy as np

sys.path.append('../..')
import pandas as pd


# Checks sessions if ADDED_PRODUCT_TO_CART is at the end
def check_sessions(df):
    df.sort_values(by=['session_id', 'created_at'], inplace=True)

    # Function to check the last action of each session
    def check_last_action(group):
        # Check if the last action in the group is 'ADDED_PRODUCT_TO_CART'
        if group['interaction_type'].iloc[-1] != 'ADDED_PRODUCT_TO_CART':
            return group['session_id'].iloc[0]  # Return session_id if not ending with 'ADDED_PRODUCT_TO_CART'

    # Apply the check to each session and collect session IDs
    non_compliant_sessions = df.groupby('session_id').apply(check_last_action).dropna()

    return non_compliant_sessions

#Splits sesssions for each session to contain only 1 ADDED_PRODUCT_TO_CART
def split_sessions_per_addition_to_cart(df):
    df = df.sort_values(by=["session_id","created_at"])
    # Count 'added_product' per session
    added_counts = df[df['interaction_type'] == 'ADDED_PRODUCT_TO_CART'].groupby('session_id').size()
    sessions_to_split = added_counts[added_counts > 1].index

    # Filter DataFrame to only those sessions that need splitting
    df_split = df[df['session_id'].isin(sessions_to_split)].copy()
    df_remain = df[~df['session_id'].isin(sessions_to_split)].copy()

    # Identify where new sessions start (after each 'added_product')
    df_split['start_new_session'] = df_split['interaction_type'].eq('ADDED_PRODUCT_TO_CART').shift(1).fillna(False).astype(bool)
    df_split['session_increment'] = df_split['start_new_session'].cumsum()

    # Assign new session IDs based on the current max session_id + increment
    max_existing_id = df['session_id'].max()
    df_split['session_id'] = max_existing_id + 1 + df_split['session_increment']

    # Combine the split and unsplit dataframes
    df_updated = pd.concat([df_remain, df_split]).sort_index()

    return df_updated.drop(columns=['start_new_session', 'session_increment'])

def truncate_sessions(df, max_length):
    # Sort DataFrame by 'session_id' and 'created_at' in ascending order for correct processing
    df.sort_values(by=['session_id', 'created_at'], ascending=[True, True], inplace=True)

    def truncate_group(group):
        # Ensure the group ends with "ADDED_PRODUCT_TO_CART"
        if group['interaction_type'].iloc[-1] != 'ADDED_PRODUCT_TO_CART':
            raise ValueError(f"Session {group['session_id'].iloc[0]} does not end with 'ADDED_PRODUCT_TO_CART'")

        # Truncate to the last 'max_length' records, keeping the session end intact
        if len(group) > max_length:
            return group.iloc[-max_length:]  # Keep only the latest 'max_length' records
        return group

    # Apply the truncation to each session
    truncated_df = df.groupby('session_id').apply(truncate_group).reset_index(drop=True)

    return truncated_df


def filter_sessions_by_duration(df, max_duration_minutes=30):
    # Calculate the time span for each session
    session_durations = df.groupby('session_id')['created_at'].agg(lambda x: x.max() - x.min())

    # Identify sessions exceeding the maximum duration
    long_sessions = session_durations[session_durations > pd.Timedelta(minutes=max_duration_minutes)].index

    # Filter out long sessions
    df_filtered = df[~df['session_id'].isin(long_sessions)]

    return df_filtered


def pre_process_raw_data(df, config):
    df = split_sessions_per_addition_to_cart(df)

    # 1) Drop row if customer_price_cz is missing - do NOT drop the final row of a session
    if "customer_price_cz" in config.model_columns:
        df["customer_price_cz"] = pd.to_numeric(df["customer_price_cz"], errors='coerce')
        # Create a mask that identifies rows with missing 'customer_price_cz' that are not 'ADDED_PRODUCT_TO_CART'
        mask = df['customer_price_cz'].isna() & (df['interaction_type'] != 'ADDED_PRODUCT_TO_CART')
        # Use the mask to filter out unwanted rows
        df = df[~mask]

    # 2a) Drop duplicates (session_id, product_variant_hash) - do NOT drop the final row of a session
    duplicate_mask_product_variant = df.duplicated(subset=['session_id', 'product_variant_hash'], keep="first")
    mask_to_keep_product_variant = (df['interaction_type'] == 'ADDED_PRODUCT_TO_CART') | ~duplicate_mask_product_variant
    df = df[mask_to_keep_product_variant]

    # 2b) Drop duplicates (session_id, product_id) - do NOT drop the final row of a session
    duplicate_mask_product_id = df.duplicated(subset=['session_id', 'product_id'], keep="first")
    mask_to_keep_product_id = (df['interaction_type'] == 'ADDED_PRODUCT_TO_CART') | ~duplicate_mask_product_id
    df = df[mask_to_keep_product_id]

    # 3a) Process quality,package_size, user_id
    if "quality" in config.model_columns:
        quality_keep_values = ['Unknown', 'Basic', 'Extra', 'Premium']
        df.loc[:, 'quality'] = np.where(df['quality'].isin(quality_keep_values), df['quality'], 'Unknown')
        df.loc[:, 'quality'] = df['quality'].fillna('Unknown')

    if "package_size" in config.model_columns:
        package_size_keep_values = ["SMALL_BOX", "BIG_BOX", "OVERSIZED_BOX", "Unknown"]
        df.loc[:, 'package_size'] = np.where(df['package_size'].isin(package_size_keep_values), df['package_size'],
                                             'Unknown')
        df.loc[:, 'package_size'] = df['package_size'].fillna('Unknown')

    df['user_id'] = df['user_id'].fillna(-1)  # Probably set by config?

    # 3b) Process srdcovky and rating
    if "pocet_srdcovky" in config.model_columns:
        df["pocet_srdcovky"] = pd.to_numeric(df["pocet_srdcovky"], errors='coerce')
        df["pocet_srdcovky"] = df["pocet_srdcovky"].fillna(0)
        df["pocet_srdcovky"] = df["pocet_srdcovky"].astype(int)

    if "rating_lifetime" in config.model_columns:
        df["rating_lifetime"] = pd.to_numeric(df["rating_lifetime"], errors='coerce', downcast="float")
        df["rating_lifetime"] = df["rating_lifetime"].fillna(0)
        df["rating_lifetime"] = df["rating_lifetime"].astype(int)

    # 4) Drop sessions based on length
    # Not to have sessions like visited product_id 1 -> bought product_id 1
    # Drop potential bots scrapings
    print("Shape of dataset before filtering large session", df.shape)
    print("Upper bound for filtering", config.max_session_length_upper_bound)
    session_lengths = df.groupby('session_id').size()
    valid_sessions = session_lengths[(session_lengths > 2) & (session_lengths <= config.max_session_length_upper_bound)].index
    df = df[df['session_id'].isin(valid_sessions)]
    print("Shape after filtering by bounds", df.shape)


    # X) Drop sessions, that are not complient ~ ADDED_PRODUCT_TO_CART is not end of the session
    non_compliant_sessions = check_sessions(df)
    print("Original shape before dropping non-compliant", df.shape)
    df = df[~df['session_id'].isin(non_compliant_sessions)]
    print("FInal shape after dropping non-compliant", df.shape)

    # 5) Truncating dataset based on max length
    df = truncate_sessions(df, config.max_length_to_truncate_to)
    print("Shape after truncation", df.shape)

    # 6) Filter
    df = filter_sessions_by_duration(df, config.max_session_duration_from_start_to_end_minutes)

    return df
