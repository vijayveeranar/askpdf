import pinecone 
from langchain.vectorstores import Pinecone
import streamlit as st
import openai
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
pinecone.init(
            api_key=st.secrets["PINECONE_API_KEY"],  # find at app.pinecone.io
            environment=st.secrets["PINECONE_API_ENV"]  # next to api key in console
        )
index_name = "bookdb" # put in the name of your pinecone index here 

index = pinecone.Index(index_name)
index_stats_response = index.describe_index_stats()
print(index_stats_response)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
docsearch = Pinecone.from_existing_index(index_name, embeddings)

def get_similiar_docs(query,k=3,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = docsearch.max_marginal_relevance_search(query,k=k, fetch_k=10)
  return similar_docs

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']
