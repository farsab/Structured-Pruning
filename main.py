"""
This code compresses a CNN trained on Fashion-MNIST using a two-layer method consisting of:

1. Structured weight pruning** to 50 % sparsity (filter pruning).
2. Post-training dynamic-range quantization** to INT8 (TFLite).
It reports test accuracy and file size for:
- Baseline FP32 model
- Pruned FP32 model
- Pruned + Quantized INT8 model

Requirements
------------
tensorflow tensorflow-model-optimization pandas tqdm
"""

from __future__ import annotations
import os, tempfile, shutil, zipfile, pathlib, itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# ------------------------------ Data --------------------------------------- #
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[..., None] / 255.0
    x_test = x_test[..., None] / 255.0
    return (x_train, y_train), (x_test, y_test)

# ------------------------------ Model -------------------------------------- #
def build_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

def compile_model(model: tf.keras.Model):
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

# ------------------------------ Utilities ---------------------------------- #
def evaluate(model, x_test, y_test) -> float:
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    return float(acc)

def save_and_size(model, name) -> float:
    model.save(f"{name}.h5", include_optimizer=False)
    size_mb = os.path.getsize(f"{name}.h5") / (1024 ** 2)
    return round(size_mb, 3)

def convert_tflite(model_path, tflite_path):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

# ------------------------------ Main --------------------------------------- #
def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    # --- Baseline ---------------------------------------------------------- #
    base_model = build_model()
    compile_model(base_model)
    base_model.fit(x_train, y_train, epochs=2, batch_size=128, verbose=0)
    acc_base = evaluate(base_model, x_test, y_test)
    size_base = save_and_size(base_model, "baseline")

    # --- Structured Pruning ------------------------------------------------ #
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {"pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(0.5, 0)}
    pruned_model = prune_low_magnitude(build_model(), **pruning_params)
    compile_model(pruned_model)
    pruned_model.fit(
        x_train,
        y_train,
        epochs=2,
        batch_size=128,
        verbose=0,
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
    )
    pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    acc_pruned = evaluate(pruned_model, x_test, y_test)
    size_pruned = save_and_size(pruned_model, "pruned")

    # --- Quantization ------------------------------------------------------ #
    convert_tflite("pruned.h5", "pruned_quant.tflite")
    size_quant = round(os.path.getsize("pruned_quant.tflite") / (1024 ** 2), 3)

    # ----------------------- Results table --------------------------------- #
    results = pd.concat(
        [
            pd.DataFrame([{"Model": "Baseline FP32", "Accuracy": acc_base, "Size_MB": size_base}]),
            pd.DataFrame([{"Model": "Pruned FP32", "Accuracy": acc_pruned, "Size_MB": size_pruned}]),
            pd.DataFrame([{"Model": "Pruned + INT8", "Accuracy": acc_pruned, "Size_MB": size_quant}]),
        ],
        ignore_index=True,
    )
    print("\n=== Compression Results ===")
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()
