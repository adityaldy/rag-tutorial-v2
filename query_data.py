import argparse
import time
# from langchain.vectorstores.chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma  # Updated import
from langchain_ollama import OllamaLLM  # Updated import

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Hanya gunakan bahasa Indonesia. Jawablah pertanyaan berdasarkan konteks berikut ini:

{context}

---

Hanya gunakan bahasa Indonesia. Jawablah pertanyaan berdasarkan konteks di atas: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    query_rag(query_text,db)


def query_rag(query_text: str,db):
    # Prepare the DB.
    # embedding_function = get_embedding_function()
    # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    #dari sini
    # Search the DB.
    start_time = time.time()  # Record start time
    results = db.similarity_search_with_score(query_text, k=5)
    end_time = time.time()  # Record end time
    query_time = end_time - start_time  # Calculate query time

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = OllamaLLM(model="mistral-nemo:latest")
    start_time = time.time()  # Record start time for LLM response
    response_text = model.invoke(prompt)
    end_time = time.time()  # Record end time for LLM response
    answer_time = end_time - start_time  # Calculate LLM response time
    #sampai sini

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}\nQuery Time: {query_time:.2f} seconds\nAnswer Time: {answer_time:.2f} seconds"
    print(formatted_response)
    # Display chunk content
    #for doc, _score in results:
    #    print(f"Chunk: {doc.page_content}")
    return response_text


if __name__ == "__main__":
    main()
