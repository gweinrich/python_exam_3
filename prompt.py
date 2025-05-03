from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pdfplumber
import pandas as pd

documents = []
for filename in os.listdir("./ENGINEERING"):
    if filename.endswith(".pdf"):
        file_path = os.path.join("./ENGINEERING", filename)
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            documents.append(text.strip())

# Initialize components
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(documents, embeddings)
llm = OpenAI(temperature=0)

# RAG prompt template
rag_template = """Context: {context}
Question: {question}
Answer:"""
rag_prompt = PromptTemplate(template=rag_template, input_variables=["context", "question"])
rag_chain = LLMChain(llm=llm, prompt=rag_prompt)
