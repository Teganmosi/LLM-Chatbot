# LLM-Chatbot
This GitHub repository contains code for performing question answering with sources using the LangChain library. The code leverages various tools and libraries, such as Google Search, newspaper3k, and OpenAI, to retrieve relevant information and generate accurate responses.


## Installation

To run this code, you need to install the required dependencies. You can install them using the following command:

```
!pip install -q langchain==0.0.208 openai tiktoken newspaper3k
```

## Setup

Before running the code, you need to set up the required API keys. Make sure to replace `<Custom_Search_Engine_ID>`, `<Google_API_Key>`, and `<OpenAI_Key>` with your respective keys.

```python
import os

os.environ["GOOGLE_CSE_ID"] = "<Custom_Search_Engine_ID>"
os.environ["GOOGLE_API_KEY"] = "<Google_API_Key>"
os.environ["OPENAI_API_KEY"] = "<OpenAI_Key>"
```

## Question Answering

The code starts by defining a question template and creating an instance of the LangChain question answering model:

```python
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

template = """You are an assistant that answers the following question correctly and honestly: {question}\n\n"""
prompt_template = PromptTemplate(input_variables=["question"], template=template)

question_chain = LLMChain(llm=llm, prompt=prompt_template)
```

It then runs a sample question to demonstrate the functionality:

```python
question_chain.run("what is the latest fast and furious movie?")
```

## Google Search Tool

The code includes a tool for performing Google searches and retrieving the top N results. It utilizes the `GoogleSearchAPIWrapper` and is defined as follows:

```python
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()
TOP_N_RESULTS = 10

def top_n_results(query):
    return search.results(query, TOP_N_RESULTS)

tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=top_n_results
)
```

You can use the tool by running:

```python
query = "what is the latest fast and furious movie?"
results = tool.run(query)

for result in results:
    print(result["title"])
    print(result["link"])
    print(result["snippet"])
    print("-" * 50)
```

## Content Extraction

The code uses the `newspaper3k` library to download and extract the text content from the URLs obtained from the search results. It stores the content in a list of dictionaries, where each dictionary contains the URL and the extracted text:

```python
import newspaper

pages_content = []

for result in results:
    try:
        article = newspaper.Article(result["link"])
        article.download()
        article.parse()
        if len(article.text) > 0:
            pages_content.append({"url": result["link"], "text": article.text})
    except:
        continue

print(len(pages_content))
```

## Text Splitting and Embedding

The code splits the extracted text into smaller chunks and embeds both the chunks and the query using OpenAI's embedding model:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

docs = []

for d in pages_content:
    chunks = text_splitter.split_text(d["text"])
    for chunk in chunks:
        new_doc = Document(page_content=chunk, metadata={"source": d["url"]})
        docs.append(new_doc)

docs_embeddings = embeddings.embed_documents([doc.page_content for doc in docs])
query_embedding = embeddings.embed_query(query)
```

## Cosine Similarity and Top K Documents

The code calculates the cosine similarity between the document vectors and the query vector using numpy and sklearn. It retrieves the top K indices and selects the corresponding best documents:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_top_k_indices(list_of_doc_vectors, query_vector, top_k):
    # Convert the lists of vectors to numpy arrays
    list_of_doc_vectors = np.array(list_of_doc_vectors)
    query_vector = np.array(query_vector)

    # Compute cosine similarities
    similarities = cosine_similarity(query_vector.reshape(1, -1), list_of_doc_vectors).flatten()

    # Sort the vectors based on cosine similarity
    sorted_indices = np.argsort(similarities)[::-1]

    # Retrieve the top K indices from the sorted list
    top_k_indices = sorted_indices[:top_k]

    return top_k_indices

top_k = 2
best_indexes = get_top_k_indices(docs_embeddings, query_embedding, top_k)
best_k_documents = [doc for i, doc in enumerate(docs) if i in best_indexes]
```

## Question Answering with Sources

The code loads a question answering chain that incorporates sources using the `load_qa_with_sources_chain` function:

```python
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")
```

Finally, it generates a response to the query and prints the answer and sources:

```python
response = chain({"input_documents": best_k_documents, "question": query}, return_only_outputs=True)

response_text, response_sources = response["output_text"].split("SOURCES:")
response_text = response_text.strip()
response_sources = response_sources.strip()

print(f"Answer: {response_text}")
print(f"Sources: {response_sources}")
```

This repository provides a comprehensive example of using LangChain to perform question answering with sources, utilizing tools like Google Search and newspaper3k for information retrieval and processing.
