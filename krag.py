import spacy
from spacy.tokens import DocBin
from spacy.util import minibatch, compounding
import random
import networkx as nx
from langchain.chains import LLMChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pdfplumber
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from spacy.training import Example

def train_ner_model(train_data, iterations=30):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                nlp.update(examples, drop=0.5, losses=losses)
            print(f"Iteration {itn}, Losses: {losses}")
    return nlp


def extract_relationships(doc):
    relationships = []
    for sent in doc.sents:
        root = sent.root
        subject = None
        obj = None
        for child in root.children:
            if child.dep_ == "nsubj":
                subject = child
            if child.dep_ in ["dobj", "pobj"]:
                obj = child
        if subject and obj:
            relationships.append((subject, root, obj))
    return relationships

def process_document(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    relationships = extract_relationships(doc)
    return entities, relationships

def build_knowledge_graph(documents):
    G = nx.DiGraph()
    for doc in documents:
        entities, relationships = process_document(doc)
        for entity, entity_type in entities:
            G.add_node(entity, type=entity_type)
        for subj, pred, obj in relationships:
            G.add_edge(subj.text, obj.text, relation=pred.text)
    return G

def rag_query(question, k=3):
    relevant_docs = vectorstore.similarity_search(question, k=k)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    return rag_chain.run(context=context, question=question)

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

# # Load dataset and remove unnecessary columns
# df_data = pd.read_json("./Entity Recognition in Resumes.json", lines=True)
# df_data = df_data.drop(['extras'], axis=1)

# # Replace newline characters for uniformity
# df_data['content'] = df_data['content'].str.replace("\n", " ")

# # Display a sample of the dataset
# df_data.head()

# train_data = []
# for _, row in df_data.iterrows():
#     text = row["content"]
#     raw_entities = row["annotation"]  # Likely a list of dicts with start, end, label

#     formatted_entities = []
#     for ent in raw_entities:
#         if isinstance(ent, dict) and "start" in ent and "end" in ent and "label" in ent:
#             formatted_entities.append((ent["start"], ent["end"], ent["label"]))
#         elif isinstance(ent, (list, tuple)) and len(ent) == 3:
#             formatted_entities.append(tuple(ent))  # already valid
#         else:
#             print("Skipping malformed entity:", ent)

#     train_data.append((text, {"entities": formatted_entities}))

# # Train the model with your domain-specific data
# ner_model = train_ner_model(train_data)
# ner_model.to_disk("./domain_ner_model")

nlp = spacy.load("./domain_ner_model")
nlp.add_pipe("sentencizer")

documents = []
for filename in os.listdir("./ENGINEERING"):
    if filename.endswith(".pdf"):
        file_path = os.path.join("./ENGINEERING", filename)
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            documents.append(text.strip())

knowledge_graph = build_knowledge_graph(documents)

# Initialize components
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(documents, embeddings)
llm = OpenAI(temperature=0)

# RAG prompt template
# Define a test question
question = "What engineering experience is described in the documents?"

# Run KRAG query instead of plain RAG
result = krag_query(question)

# Print the result
print(result)

print(llm.invoke("Say hello"))
