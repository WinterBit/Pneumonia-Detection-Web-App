import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model (update path if needed)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),  # Input shape matches the resized images
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Optional: Load weights if pre-trained
# model.load_weights('model.h5')  # Uncomment this line if you have a pre-trained model

# App title and description
st.title('Pneumonia Detection Web App')
st.write('Upload an image of a chest X-ray to predict pneumonia.')

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Open the image using PIL
    img = Image.open(uploaded_image)

    # Display the uploaded image
    st.image(img, caption='Uploaded Image.', use_container_width=True)

    # Resize the image to (150, 150) as required by the model
    img = img.resize((150, 150))

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Check if the image has 3 channels (RGB), and if not, convert it
    if img_array.ndim == 2:  # Grayscale image
        img_array = np.stack([img_array] * 3, axis=-1)  # Convert grayscale to RGB by duplicating the channel

    # Ensure the image has 3 channels
    if img_array.shape[-1] != 3:
        raise ValueError("Input image must have 3 channels (RGB)")

    # Add an extra dimension for batch size: shape becomes (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image (assuming model expects values in the range [0, 1])
    img_array = img_array / 255.0

    # Make a prediction using the model
    prediction = model.predict(img_array)

    # Assuming a binary classification model: 0 = no pneumonia, 1 = pneumonia
    pneumonia_probability = prediction[0][0]  # For binary classification, output is a single value

    # Convert the probability to a percentage
    pneumonia_percentage = pneumonia_probability * 100

    # Display the result as a percentage
    st.write(f"Probability of Pneumonia: {pneumonia_percentage:.2f}%")

    # Basic decision based on the probability (You can modify the threshold as per your model)
    if pneumonia_percentage > 50:
        st.write("The image is likely to show pneumonia.")
    else:
        st.write("The image is likely to show no pneumonia.")
