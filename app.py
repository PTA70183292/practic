import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Анализ тональности")

@st.cache_resource
def load_model():
    model_path = "./results/BERT_QLoRA"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

st.title("Анализ тональности")

try:
    tokenizer, model = load_model()
    st.success("Модель загружена")
except:
    st.error("Модель не найдена. Убедитесь, что модель обучена.")
    st.stop()

text_input = st.text_area("Введите текст:", height=150)

if st.button("Анализировать"):
    if text_input:
        st.info("Анализ...")
    else:
        st.warning("Пожалуйста, введите текст")
