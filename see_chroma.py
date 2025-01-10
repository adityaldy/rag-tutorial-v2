from langchain_chroma import Chroma  # Updated import

# Replace with your actual Chroma database directory
CHROMA_PATH = "chroma"

# Initialize ChromaDB
db = Chroma(persist_directory=CHROMA_PATH)

# Get all documents
results = db.get(include=["documents"])  # Corrected line

# Print the documents and IDs
for doc in results["documents"]:
    print(f"Document Content: {doc}")
    print("---")