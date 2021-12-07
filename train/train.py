logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

def build_cnn(seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            8, (4, 3),
            padding="same",
            activation="relu",
            input_shape=(seq_length, 3, 1)
        ),
        tf.keras.layers.MaxPool2D((3, 3)),
        tf.keras.layers.Droppout(0.1),
        tf.keras.layers.Conv2D(16, (4, 1), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D((3, 1), padding="same"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(9, activation="softmax")
    ])
    model_path = os.path.join("./models", "CNN")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.load_weights(os.path.join(model_path, "weights.h5"))
    return model, model_path

def load_data(train_data_path, valid_data_path, test_data_path, seq_length):
    data_loader = DataLoader(
        train_data_path,
        valid_data_path,
        test_data_path,
        seq_length=seq_length
    )
    data_loader.format()
    return data_loader.train_len, data_loader.train_data, data_loader.valid_len, \
    data_loader.valid_data, data_loader.test_len, data_loader.test_data

if __name__ == "__main__":
    seq_length = 128
    data = load_data("./data/motion_data_22_users.csv")

    