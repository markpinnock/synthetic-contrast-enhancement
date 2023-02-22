import argparse
import datetime
import os
from pathlib import Path
import tensorflow as tf
import yaml

from multi_phase.networks.models import get_model
from multi_phase.trainingloops.build_training_loop import get_training_loop
from multi_phase.utils.build_dataloader import get_train_dataloader


# -------------------------------------------------------------------------


def train(config: dict):
    # Get datasets and data generator
    train_ds, val_ds, train_gen, val_gen = get_train_dataloader(config)

    # Compile model
    Model = get_model(config)

    if config["expt"]["verbose"]:
        Model.summary()

    # Write graph for visualising in Tensorboard
    if config["expt"]["graph"]:
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"{config['paths']['expt_path']}/logs/{curr_time}"
        writer = tf.summary.create_file_writer(log_dir)

        @tf.function
        def trace(x):
            return Model.Generator(x)

        tf.summary.trace_on(graph=True)
        trace(tf.zeros([1] + config["hyperparameters"]["img_dims"] + [1]))

        with writer.as_default():
            tf.summary.trace_export("graph", step=0)

    TrainingLoop = get_training_loop(
        Model=Model,
        dataset=(train_ds, val_ds),
        train_generator=train_gen,
        val_generator=val_gen,
        config=config,
    )

    # Run training loop
    TrainingLoop.train()


# -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--gpu", "-g", help="GPU number", type=int)
    arguments = parser.parse_args()

    expt_path = Path(arguments.path)
    (expt_path / "images").mkdir(exist_ok=True)
    (expt_path / "logs").mkdir(exist_ok=True)
    (expt_path / "models").mkdir(exist_ok=True)

    # Parse config json
    with open(expt_path / "config.yml", 'r') as fp:
        config = yaml.load(fp, yaml.FullLoader)

    config["paths"]["expt_path"] = arguments.path

    # Set GPU
    if arguments.gpu is not None:
        gpu_number = arguments.gpu
        os.environ["LD_LIBRARY_PATH"] = config["paths"]["cuda_path"]
        gpus = tf.config.experimental.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[gpu_number], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_number], True)

    train(config)


# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()
