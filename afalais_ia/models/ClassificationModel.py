from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import tensorflow as tf
IMG_WIDTH = 96
IMG_HEIGHT = 96
NUM_CLASSES = 3


def get_classification_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation="relu",
                               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="sigmoid"),
        tf.keras.layers.Dense(units=NUM_CLASSES)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=tf.float32)])

    return model
