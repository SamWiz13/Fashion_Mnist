import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image
import numpy as np

st.cache_data()
with st.spinner('Model is being loaded..'):
    model = keras.models.load_model('fashion_mnist.h5')
file = st.file_uploader("Please upload a fashion image file", type=["jpg", "jpeg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    # Resize image to (28, 28) while preserving aspect ratio
    image = image.resize((28, 28))

    # Convert image to grayscale numpy array and normalize pixel values
    image_array = np.array(image.convert("L"))
    image_array = image_array.astype('float32') / 255.0

    # Reshape image array to match expected input shape of model
    image_array = np.reshape(image_array, (1, 28, 28, 1))

    # Predict class probabilities using model and get predicted class label
    class_probs = model.predict(image_array)[0]
    predicted_class_index = np.argmax(class_probs)
    predicted_class_label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'][predicted_class_index]

    # Display predicted class label and corresponding probability
    st.write("Predicted class:", predicted_class_label)
    st.write("Probability:", class_probs[predicted_class_index])
