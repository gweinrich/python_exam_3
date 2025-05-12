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

if __name__ == "__main__":
    try:
        print("Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm")  # Use a full pipeline that includes parser
        ner = nlp.get_pipe("ner")

        # Optionally load your domain-specific NER model weights if needed
        # Or add additional NER labels manually like before

        # Add sentencizer if it's missing
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer", before="ner")

        print(nlp.pipe_names)
        test_doc = nlp("John worked as a mechanical engineer at XYZ Corp.")
        print([(ent.text, ent.label_) for ent in test_doc.ents])

        print("Loading documents...")
        documents = []
        for filename in os.listdir("./ENGINEERING"):
            if filename.endswith(".pdf"):
                file_path = os.path.join("./ENGINEERING", filename)
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    documents.append(text.strip())
        print(f"Loaded {len(documents)} documents.")
        doc = nlp(documents[0])
        print(f"Number of sentences: {len(list(doc.sents))}")
        print(f"Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")


        print("Building knowledge graph...")
        knowledge_graph = build_knowledge_graph(documents)
        print("Knowledge graph built.")

        print("Initializing embeddings and LLM...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(documents, embeddings)
        llm = OpenAI(temperature=0)

        print("Running KRAG query...")
        question = "Which applicants have experience in SQL?"
        relevant_docs = vectorstore.similarity_search(question, k=1)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        relevant_triples = get_relevant_triples(question, knowledge_graph)
        triples_context = "\n".join(relevant_triples)

        krag_template = """Context: {context}
Relevant Knowledge Graph Triples:
{triples_context}
Question: {question}
Answer:"""

        krag_prompt = PromptTemplate(
            template=krag_template,
            input_variables=["context", "triples_context", "question"]
        )
        krag_chain = LLMChain(llm=llm, prompt=krag_prompt)
        result = krag_chain.run(context=context, triples_context=triples_context, question=question)

        print("Result:", result)

    except Exception as e:
        import traceback
        print("An error occurred:")
        traceback.print_exc()
