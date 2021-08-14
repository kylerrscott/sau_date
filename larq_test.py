import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import larq_zoo as lqz
from urllib.request import urlopen
from PIL import Image

img_path = "https://raw.githubusercontent.com/larq/zoo/master/tests/fixtures/elephant.jpg"

with urlopen(img_path) as f:
    img = Image.open(f).resize((224, 224))

x = tf.keras.preprocessing.image.img_to_array(img)
x = lqz.preprocess_input(x)
x = np.expand_dims(x, axis=0)

model = lqz.sota.QuickNet(weights="imagenet")
preds = model.predict(x)
lqz.decode_predictions(preds, top=5)[0]

tf.keras.backend.clear_session()
model = lqz.sota.QuickNet(weights="imagenet", include_top=False)
features = model.predict(x)
print("Feature shape:", features.shape)

def preprocess(data):
    img = lqz.preprocess_input(data["image"])
    label = tf.one_hot(data["label"], 1000)
    return img, label 

dataset = (
    tfds.load("imagenet2012:5.0.0", split=tfds.Split.VALIDATION)
    .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(128)
    .prefetch(1)
)

model = lqz.sota.QuickNet()
model.compile(
    optimizer="sgd",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy", "top_k_categorical_accuracy"],
)

model.evaluate(dataset)
