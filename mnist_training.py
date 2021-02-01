import tensorflow as tf
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config")
def training(cfg: DictConfig) -> None:

    mlflow.set_experiment("test")

    #  mlflow.set_tracking_uri("file:/media/guillaume/Data/mlflow_logs")

    print(OmegaConf.to_yaml(cfg))

    important_params = ['hidden_dim', 'dropout']
    for param in important_params:
        mlflow.log_param(param, cfg['param'])

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(cfg.hidden_dim, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    class LogMetricsCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            mlflow.log_metric('training_loss', logs['loss'], epoch)
            mlflow.log_metric('training_accuracy', logs['accuracy'], epoch)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, callbacks=[LogMetricsCallback()])
    mlflow.keras.log_model(keras_model=model, artifact_path='model')


if __name__ == "__main__":

    training()



