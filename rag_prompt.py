from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pdfplumber
import pandas as pd
import psutil
import time
from dotenv import load_dotenv

process = psutil.Process(os.getpid())

start_time = time.time()
start_cpu = process.cpu_percent(interval=None)
start_mem = process.memory_info().rss
    
load_dotenv()

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
relevant_docs = vectorstore.similarity_search(question, k=4)

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

end_time = time.time()
end_cpu = process.cpu_percent(interval=None)
end_mem = process.memory_info().rss  # in bytes

print(f"Execution Time: {end_time - start_time:.4f} seconds")
print(f"CPU Usage: {end_cpu - start_cpu:.2f}%")
print(f"Memory Usage: {(end_mem - start_mem) / 1024 / 1024:.4f} MB")
