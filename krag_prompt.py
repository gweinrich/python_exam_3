from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pdfplumber
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def get_relevant_triples(question, graph, k=5):
    entities = nlp(question).ents
    relevant_triples = []
    for entity in entities:
        if entity.text in graph:
            neighbors = list(graph.neighbors(entity.text))
            for neighbor in neighbors[:k]:
                edge_data = graph.get_edge_data(entity.text, neighbor)
                relevant_triples.append(f"{entity.text} {edge_data['relation']} {neighbor}")
    return relevant_triples

def krag_query(question, k=3):
    relevant_docs = vectorstore.similarity_search(question, k=k)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    relevant_triples = get_relevant_triples(question, knowledge_graph)
    triples_context = "\n".join(relevant_triples)
    krag_template = """Context: {context}
    Relevant Knowledge Graph Triples:
    {triples_context}
    Question: {question}
    Answer:"""
    krag_prompt = PromptTemplate(template=krag_template,
    input_variables=["context", "triples_context", "question"])
    krag_chain = LLMChain(llm=llm, prompt=krag_prompt)
    return krag_chain.run(context=context, triples_context=triples_context, question=question)


# Step 1: Load PDF documents
documents = []
for filename in os.listdir("./ENGINEERING"):
    if filename.endswith(".pdf"):
        file_path = os.path.join("./ENGINEERING", filename)
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
            documents.append(text.strip())

# Step 2: Initialize vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(documents, embeddings)

# Step 3: Define a user question
question = "What engineering experience is described in the documents?"

# Step 4: Perform similarity search to get relevant documents
relevant_docs = vectorstore.similarity_search(question, k=3)

# Step 5: Extract context from relevant documents
context = "\n".join([doc.page_content for doc in relevant_docs])

# Step 6: Initialize LLM
llm = OpenAI(temperature=0)

# Step 7: Define prompt and RAG chain
rag_template = """Context: {context}
Question: {question}
Answer:"""

rag_prompt = PromptTemplate(template=rag_template, input_variables=["context", "question"])
rag_chain = rag_prompt | llm

# Step 8: Run the chain
result = rag_chain.invoke({"context": context, "question": question})

# Output the result
print(result)
