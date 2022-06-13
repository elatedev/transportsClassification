import streamlit as st 
from fastai.vision.all import * 
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
    
st.title("Model that can classify pictures of transports. (Boats, Cars, Airplanes)")

# uploading an image

file = st.file_uploader('Rasm Yuklash', type=['png', 'jpeg', 'gif', 'svg', 'jpg'])

if file:
    # model
    model = load_learner('transport_model.pkl')
    st.image(file)
    img = PILImage.create(file)
    # prediction
    pred, pred_id, probs = model.predict(img)

    st.success(f"Prediction: {pred}")
    st.info(f"Accuracy: {probs[pred_id]*100:.1f}%")

    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
