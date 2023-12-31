import streamlit as st 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import chromadb



#model and tokenizer loading
checkpoint = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.float32)


@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 384,
        do_sample=True,
        temperature = 0.3,
        top_p = 0.95
    )
    local_llm = HuggingFacePipeline(pipeline = pipe)
    return local_llm


@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path = "db")
    db = Chroma(client=client, embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer,  generated_text




#sidebar Contents
with st.sidebar:
    st.title(" 📚PDF✧˖°LORE💭 ")
    st.markdown('''
    # About 
    This app is LLM-powered Q/A BOT built using:
    [Streamlit](https://streamlit.io/)
    [Langchain](https://www.langchain.com/)
    [LaMini-T5-738M](https://huggingface.co/MBZUAI/LaMini-T5-738M)

    ''')
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("---")
    st.write("Made with 🖤 by [「 ✦ livend ✦ 」](https://www.linkedin.com/in/mdfahadullahutsho/)")




def main():
    st.header("🧙‍♂️Probe your PDF for the answers you seek")
    with st.expander("About the App"):
        st.markdown(
            """
            This is a Generative AI powered Question and Answering app that responds to questions about your PDF File.
            """
        )
    question = st.text_area("Enter your Question")
    if st.button("ASK"):
        st.info("Your Question: " + question)

        
        answer, metadata = process_answer(question)
        st.info(answer)
        st.header("😎🤏😳🕶🤏")
        st.write(metadata)


if __name__ == '__main__':
    main()