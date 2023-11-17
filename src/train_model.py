import tensorflow as tf
import os
import numpy as np
import data
from CNN_Model import CNN_Model

data_loader = data.Data("/home/vyas/CVIP/project/Dataset")
INPUT_IMAGE_DIR = data_loader.input_path
OUTPUT_IMAGE_DIR = data_loader.output_path

class Train():
    def __init__(self):
        self.input_path = INPUT_IMAGE_DIR
        self.output_path = OUTPUT_IMAGE_DIR


    def load_and_preprocess_image(self, path, channels):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=channels)
        image = tf.image.resize(image, [128, 128])
        image = (image / 255.0) 
        return image

    def create_dataset(self, input_dir, output_dir):
        input_paths = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir)]
        output_paths = [os.path.join(output_dir, fname) for fname in os.listdir(output_dir)]
    
        dataset = tf.data.Dataset.from_tensor_slices((input_paths, output_paths))
    
        dataset = dataset.map(lambda inp, out: (
            self.load_and_preprocess_image(inp, channels=3),
            self.load_and_preprocess_image(out, channels=1))) 
    
        dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)
    
        return dataset



obj_train = Train()
dataset = obj_train.create_dataset(INPUT_IMAGE_DIR, OUTPUT_IMAGE_DIR)

model = CNN_Model()

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(dataset, epochs=10)


model.predict("/home/vyas/CVIP/project/Dataset/bedroom_rgb_00000.jpg")






