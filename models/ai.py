import tensorflow as tf
from tensorflow import keras

class AIModel:
  def __init__(self):
    self.model = keras.models.load_model("./models/model.h5")

  def predict(self, img_path: str):
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224, 3))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = self.model.predict(img_array)
    return predictions
  
model = AIModel()