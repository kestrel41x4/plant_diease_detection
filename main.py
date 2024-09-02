import streamlit as st
import tensorflow as tf
import numpy as np


def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index



st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home","About", "Disease Recognition"])



if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.markdown("""
        This is a project designed to find diseases in plants. 
        Upload a pic and we'll identify it
    """)

elif(app_mode=="About"):
    st.header("about")
    st.markdown("""
        This is the data set
    """)

elif(app_mode=="Disease Recognition"):
    st.header("Diease Recognition")
    test_image = st.file_uploader("Choose an Image")

    if(st.button("Show Image")):
        st.image(test_image, use_column_width=True)
    if(st.button("Predict")):
        with st.spinner("please wait"):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            class_name = ['Apple - Apple Scab',
                          'Apple - Black Rot',
                          'Apple - Cedar_apple Rust',
                          'Apple - Healthy',
                          'Blueberry - Healthy',
                          'Cherry - Powdery Mildew',
                          'Cherry - healthy',
                          'Corn - Cercospora Leaf Spot Gray Leaf Spot',
                          'Corn - Common_rust_',
                          'Corn - Northern_Leaf_Blight',
                          'Corn - healthy',
                          'Corn - Black_rot',
                          'Grape - Black_Measles',
                          'Grape - Leaf_blight',
                          'Grape - healthy',
                          'Orange - Haunglongbing',
                          'Peach - Bacterial_spot',
                          'Peach - healthy',
                          'Bell Pepper - Bacterial_spot',
                          'Bell Peper - healthy',
                          'Potato - Early_blight',
                          'Potato - Late_blight',
                          'Potato - healthy',
                          'Raspberry - healthy',
                          'Soybean - healthy',
                          'Squash  - Powdery_mildew',
                          'Strawberry - Leaf_scorch',
                          'Strawberry - healthy',
                          'Tomato - Bacterial_spot',
                          'Tomato - Early_blight',
                          'Tomato - Late_blight',
                          'Tomato - Leaf_Mold',
                          'Tomato - Septoria_leaf_spot',
                          'Tomato - Spider_mites Two-spotted_spider_mite',
                          'Tomato - Target_Spot',
                          'Tomato - Tomato_Yellow_Leaf_Curl_Virus',
                          'Tomato - Tomato_mosaic_virus',
                          'Tomato - healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))