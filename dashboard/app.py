"""Streamlit dashboard. Run from the repo root: streamlit run dashboard/app.py"""
import os, sys
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from serving import DEFAULT_BUNDLE, Service, render

st.set_page_config(page_title='DermaAI', layout='wide')


@st.cache_resource
def get_service():
    return Service(os.environ.get('DERMAAI_BUNDLE', str(DEFAULT_BUNDLE)))


st.title('DermaAI - dermoscopic lesion decision support')
st.warning('Research prototype. This is not a medical device and not a diagnosis. '
           'Only a qualified clinician can diagnose a skin lesion.')
svc = get_service()
explain = st.checkbox('Show lesion outline (U-Net) and Grad-CAM heatmap', value=True)
file = st.file_uploader('Upload a dermoscopic image', type=['jpg', 'jpeg', 'png'])

if file:
    pil = Image.open(file).convert('RGB')
    r = svc.predict(pil, explain=explain)
    if r['referred_for_review']:
        st.error(f"Refer for expert review: confidence {r['confidence']:.0%} is below the referral threshold.")
    else:
        st.success(f"Confident prediction: {r['predicted_class']} ({r['confidence']:.0%})")
    st.write(r['message'])
    imgs = render(pil, r) if explain else {}
    cols = st.columns(3)
    cols[0].image(pil, caption='Input', use_container_width=True)
    if 'mask' in imgs:
        cols[1].image(imgs['mask'], caption='Lesion outline (context only)', use_container_width=True)
    if 'heatmap' in imgs:
        cols[2].image(imgs['heatmap'], caption='Grad-CAM (qualitative aid)', use_container_width=True)
    st.subheader('Calibrated class probabilities')
    st.bar_chart(pd.Series(r['probabilities']))
