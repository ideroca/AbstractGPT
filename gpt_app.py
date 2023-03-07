from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
from selenium import webdriver
import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken
from tqdm import tqdm
import pyAesCrypt
import json
import streamlit as st
from streamlit_chat import message



load_dotenv()

openai.api_key = st.secrets['OPENAI_API_KEY']
password = st.secrets['DECRYPT_KEY']

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"




# if ~os.path.isfile('embeddings.json'):
#     pyAesCrypt.decryptFile("embeddings.aes", "AASLD_embeddings.json", password)
#     with open("AASLD_embeddings.json", "r") as fr:
#         embs = json.load(fr)
# else:
with open("embeddings.json", "r") as fr:
    embs = json.load(fr)

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

MAX_SECTION_LEN = 700
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

task = st.radio('Select a task:', ['Question and Answer', 'Summarization'])


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df[df['title'] == section_index]
        
        chosen_sections_len += len(document_section.content.values[0].split(' ')) + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.values[0].replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    st.write(f"Selected {len(chosen_sections)} document sections:")
    st.write("\n".join(chosen_sections_indexes))

    if task == 'Question and Answer':
        header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
        return header + "".join(chosen_sections)+ "\n\nQ: " + question + "\n\nA: "
    elif task == 'Summarization':
        header = """Summarize each of the medical abstracts provided below into a bulleted list that explains the purpose, methods, and results."\n\nContext:\n"""
        return header + "".join(chosen_sections) + "\n\n Summary: "# + question + "\n A:"
    #return header + "".join(chosen_sections) + "\n\n Summary: "# + question + "\n A:"
COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 1000,
    "model": COMPLETIONS_MODEL,
}

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    # if show_prompt:
    #     print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return prompt, response["choices"][0]["text"].strip(" \n")

df = pd.read_csv('AASLD_abstracs.csv')

def chatgpt_clone(input, history):
    history = history or []
    s = list(sum(history, ()))
    #print(s)
    s.append(input)
    inp = ' '.join(s)
    prompt, output = answer_query_with_context(inp, df, embs)
    #history.append((input, f'{prompt} {output}'))
    #return history, history
    return prompt, output


# Streamlit App
# st.set_page_config(
#     page_title="Streamlit Chat - Demo",
#     page_icon=":robot:"
# )

st.header("AASLD Abstracts Q/A or Summarization")

history_input = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text


user_input = get_text()


if user_input:
    output = chatgpt_clone(user_input,history_input)
    st.markdown(f'### Prompt/Context: \n\n{output[0]}')
    st.markdown(f'### GPT generated response')
    st.write(output[1])
    # output = chatgpt_clone(user_input, history_input)
    # history_input.append([user_input, output])
    # st.session_state.past.append(user_input)
    # st.session_state.generated.append(output[1])

# if st.session_state['generated']:

#     for i in range(len(st.session_state['generated'])-1, -1, -1):
#         st.write(st.session_state["generated"][i])
#         st.markdown('\n\n')
#         st.write(st.session_state['past'][i])
#         st.markdown('\n\n')