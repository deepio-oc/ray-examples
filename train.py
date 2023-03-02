from tensorflow.keras import datasets, layers, models
import tensorflow as tf

TRAINED_MODEL_PATH = "./mnist_model.h5"

def train_and_save_model():
    # Load mnist dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Train a simple neural net model

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(10, activation='softmax'))

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=3)

    model.evaluate(x_test, y_test, verbose=1)
    model.summary()

    # Save the model in h5 format in local file system
    model.save(TRAINED_MODEL_PATH)
    
if __name__ == "__main__":
    train_and_save_model()