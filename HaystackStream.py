import streamlit as st
import pandas as pd
import random
import torch
import base64
from haystack.document_stores import InMemoryDocumentStore
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline
from haystack.utils import print_answers
from haystack import Document

import subprocess

def install_libraries():
    subprocess.run(['pip', 'install', 'transformers'])
    subprocess.run(['pip', 'install', 'torch'])
    subprocess.run(['pip', 'install', 'farm-haystack[faiss]'])
    subprocess.run(['pip', 'install', 'farm-haystack'])
    subprocess.run(['pip', 'install', 'wrapt'])

def generate_questions(context, num_questions):
    generator = pipeline('text2text-generation', model='voidful/context-only-question-generator')
    model_checkpoint = "consciousAI/question-answering-generative-t5-v1-base-s-q-c"
    device = torch.device("cpu")
    
    def answer_generate(query, context, model, device):
        FT_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)
        FT_MODEL_TOKENIZER = AutoTokenizer.from_pretrained(model)
        input_text = "question: " + query + "</s> question_context: " + context
        input_tokenized = FT_MODEL_TOKENIZER.encode(input_text, return_tensors='pt', truncation=True,
                                                    padding='max_length', max_length=1024).to(device)
        _tok_count_assessment = FT_MODEL_TOKENIZER.encode(input_text, return_tensors='pt', truncation=True).to(device)
        summary_ids = FT_MODEL.generate(input_tokenized, max_length=30, min_length=5, num_beams=2,
                                        early_stopping=True)
        output = [FT_MODEL_TOKENIZER.decode(id, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                  for id in summary_ids]
        return str(output[0])

    df = pd.DataFrame(columns=['Questions', 'Answers'])

    for i in range(num_questions):
        randi = random.uniform(0, 1.5)
        questions = generator(context, temperature=randi, do_sample=True)
        generated_question = questions[0]['generated_text']
        generated_answer = answer_generate(generated_question, context,
                                           model="consciousAI/question-answering-generative-t5-v1-base-s-q-c",
                                           device=device)
        df.loc[i] = [generated_question, generated_answer]

    return df


def generate_haystack_answers(df, context, generator, retriever):
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True, sql_url="sqlite://")
    retriever = EmbeddingRetriever(document_store=document_store,
                                   embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
                                   model_format="sentence_transformers")

    generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")

    document = Document(content=context, meta={"name": "Context"})
    documents = [document]
    document_store.delete_documents()
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever=retriever)

    pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)

    for index, row in df.iterrows():
        question = row['Questions']
        res = pipe.run(query=question, params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 5}})
        generated_answer = res['answers'][0].answer
        df.loc[index, 'Haystack answers'] = generated_answer

    return df


def main():
    st.title("Question Generation and Answering")
    st.subheader("Generate Questions and Answers")

    context = st.text_area("Enter the context:")
    num_questions = st.selectbox("Number of Questions to Generate:", options=[1, 2, 3, 4, 5], index=0)

    if st.button("Generate Questions, Answers, and Download Excel"):
        df = generate_questions(context, num_questions)
        document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True, sql_url="sqlite://")
        generator = pipeline('text2text-generation', model='voidful/context-only-question-generator')
        retriever = EmbeddingRetriever(document_store=document_store,
                                       embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
                                       model_format="sentence_transformers")
        df = generate_haystack_answers(df, context, generator, retriever)
        st.write(df)

        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="question_answer.csv">Download Excel</a>'
        st.markdown(href, unsafe_allow_html=True)


if __name__ == '__main__':
    #install_libraries()
    main()
