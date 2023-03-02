
import os
from datetime import date

import streamlit as st
import cv2
import numpy as np

from keras.models import load_model
import keras.utils as image


# paths
image_path = "image"
model_path = 'model'
file_timestamp = str(date.today().strftime("%Y_%m_%d"))

st.title('Flower predictor ')

st.text('Supported flowers: daisy, dandelion, rose, sunflower, tulip')

uploaded_file = st.file_uploader("Upload Image")


if uploaded_file is None:
    st.error('File uploaded cant be processed')

else:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")

    # save the image
    cv2.imwrite(os.path.join(image_path, "SavedImage.jpg"), opencv_image)

    categories = ['daisy','dandelion','rose','sunflower','tulip']

    # dimensions of our images
    img_width, img_height = 128, 128

    # load the model we saved
    model = load_model(os.path.join(model_path, "model.h5"))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # predicting images
    img = image.load_img(os.path.join(image_path, "SavedImage.jpg"), target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    st.title('Predicted Flower')
    st.text(categories[np.argmax(model.predict(images))])

    #retrain model
    predict_wrong = st.checkbox('Is the prediction wrong?')

    if predict_wrong:

        flower_name = st.selectbox("Name of flower", ('daisy','dandelion','rose','sunflower','tulip'))
        save_button = st.checkbox('Save')
        if save_button:
            data_load_state = st.text("saving data ....")
            file_storage_path=os.path.join("Retraining_data",flower_name)
            if not os.path.exists(file_storage_path):
                os.mkdir(os.path.join(file_storage_path))
            cv2.imwrite(os.path.join(file_storage_path, str(flower_name)+"_"+str(file_timestamp)+".jpg"), opencv_image)
            st.write(file_storage_path)




