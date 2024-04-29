import numpy as np
import pandas as pd


# Try without the anonymous users?
def train_test_split(df, encoders):
    encoded_anonymous_id = encoders['user_id'].transform([[-1]])[0][0]  # Adding [0][0] to get a single scalar value

    # Extract all sessions for anonymous users and ensure each session has more than one entry
    anonymous_sessions = df[df['user_id'] == encoded_anonymous_id]
    grouped_sessions = anonymous_sessions.groupby('session_id').filter(lambda x: len(x) > 1)

    # List of unique session IDs to handle complete sessions
    session_ids = grouped_sessions['session_id'].unique()

    # Calculate the index to split at 10% from the end
    num_sessions = len(session_ids)
    split_index = int(num_sessions * 0.9)  # 90% for training, last 10% for testing

    # Split session IDs into training and testing
    train_session_ids = session_ids[:split_index]
    test_session_ids = session_ids[split_index:]

    anonymous_test_sessions = grouped_sessions[grouped_sessions['session_id'].isin(test_session_ids)]


    regular_sessions = df[df['user_id'] != encoded_anonymous_id]
    regular_last_sessions = regular_sessions[
        regular_sessions['session_id'] == regular_sessions.groupby('user_id')['session_id'].transform('max')]
    regular_last_sessions = regular_last_sessions.groupby('session_id').filter(lambda x: len(x) > 1)

    # Create the final test and train sets
    test_sessions = pd.concat([regular_last_sessions, anonymous_test_sessions])

    train_sessions = df[~df.index.isin(test_sessions.index)]

    grouped_train_df = {name: group.reset_index(drop=True) for name, group in train_sessions.groupby('session_id')}
    grouped_test_df = {name: group.reset_index(drop=True) for name, group in test_sessions.groupby('session_id')}

    train_list = list(grouped_train_df.values())
    test_list = list(grouped_test_df.values())

    return train_list, test_list