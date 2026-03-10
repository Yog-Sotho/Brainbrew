import streamlit as st

from orchestrator import run_distillation
from config import DistillationConfig, QualityMode

st.title("Teacher's Lounge")

teacher = st.text_input("Teacher Model")

dataset_size = st.slider("Dataset Size",100,20000,2000)

quality = st.selectbox(
"Quality Mode",
["fast","balanced","research"]
)

train_model = st.checkbox("Train model automatically")

publish = st.checkbox("Publish dataset to HuggingFace")

file = st.file_uploader("Upload source document")

run = st.button("Generate Dataset")

if run:

    with open("input.txt","wb") as f:
        f.write(file.read())

    cfg = DistillationConfig(

        teacher_model=teacher,
        dataset_size=dataset_size,
        quality_mode=QualityMode(quality),
        source_file="input.txt",
        train_model=train_model,
        publish_dataset=publish

    )

    dataset = run_distillation(cfg)

    st.success("Dataset generated")

    with open(dataset,"rb") as f:

        st.download_button(
            "Download dataset",
            f,
            file_name="dataset.jsonl"
        )
