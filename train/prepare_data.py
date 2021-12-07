import json

from sklearn.model_selection import train_test_split

def write_instance(f, idx, instance):
    activity, user, _, _, _ = idx
    instance_data = {
        "gesture": activity,
        "name": str(user),
        "accel_ms2_xyz": [],
    }

    # get all measurements for this unique motion segment
    for i, (_, row) in enumerate(instance.iterrows()):

        # the dataset was colleccted at 250 Hz so we should skip every
        # other row to downsample to 125 Hz (approx 119 Hz)
        if i % 2 != 0:
            continue
        
        data_cols = ["Ax", "Ay", "Az"]#, "GyroX", "GyroY", "GyroZ"]
        instance_data["accel_ms2_xyz"].append([
            row[col] for col in data_cols
        ])
    
    # write this instance to the train/valid/test file as json
    f.write(json.dumps(instance_data) + "\n")

def prepare_interim_data(data):
    """
    Prepares the raw user activity data as an interim format split by user
    and activity, with each such file including delimited sets of measurements
    corresponding to motion segments.
    """
    # create interim data directories for each activity type
    activities = data.Activity.unique().tolist()
    for activity in activities:
        os.makedirs(f"./data/interim/{activity}", exist_ok=True)

    # Write the motion data for each activty, user, and motion segment
    for idx, group in data.groupby(["Activity", "User"]):
        
        datapath = f"./data/interim/{idx[0]}/output_{idx[0]}_{idx[1]}.txt"        
        with open(datapath, "w") as f:

            prior_trial = (None, None)
            for i, (_, row) in enumerate(group.iterrows()):
                # skip every other row to downsample to 125 Hz (approx 119 Hz)
                if i % 2 != 0:
                    continue

                current_trial = (row["Scenerio"], row["Trial"])
                if current_trial != prior_trial:
                    f.write(f"\n-,-,-\n")
                    prior_trial = current_trial         

                f.write(f"{row['Ax']},{row['Ay']},{row['Az']}\n")



def prepare_data_by_user(df):
    """
    Prepares the dataset split by user. Since there are 22 total users in the
    dataset, 14/22 (63%), 4/22 (18%), and 4/22 (18%) are selected for the 
    training, validation, and testing sets, respectively.
    """

    instance_cols = ["Activity", "User", "Scenerio", "Trial", "Window_Number"]

    # Select random users for train, validation, and test sets
    # Train = 14/22 (63%), Valid & Test = 4/22 (18%)
    # TODO: could take varying amount of data per user into account later on
    users = pd.Series(df.User.unique()).sample(frac=1).tolist()
    train_users, valid_users, test_users = users[:14], users[14:18], users[18:22]
    df["Set"] = df.User.apply(lambda x: "train" if x in train_users else \
        "valid" if x in valid_users else "test")

    # Write the motion data for each activity, user, and motion segment (instance)
    for set_type, set_group in df.groupby("Set"):
        with open(f"./data/processed/user/{set_type}", "w") as f:
            for idx, instance in set_group.groupby(instance_cols):
                write_instance(f, idx, instance)


def prepare_data_by_instance(df):
    """
    Prepares the dataset split by instance. The user generating the data is not
    considered, thus the same user can appear in train, valid, and test sets. 
    """

    instance_cols = ["Activity", "User", "Scenerio", "Trial", "Window_Number"]

    # Split the data into train, valid, and test by instance
    instance_indices = df.groupby(instance_cols, as_index=True).count().index
    train, valid = train_test_split(instance_indices, test_size=0.3)
    valid, test = train_test_split(valid, test_size=0.5)
    sets = {"train": train, "valid": valid, "test": test}

    # Write the motion data for each activity, user, and motion segment (instance)
    for set_name, set_idx in sets.items():
        set_instances = df.set_index(instance_cols).loc[set_idx].reset_index()
        with open(f"./data/processed/user/{set_name}", "w") as f:
            for idx, instance in set_instances.groupby(instance_cols):
                write_instance(f, idx, instance)