import os
import numpy
import json

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  base_url=os.getenv("BASE_URL"),
  api_key=os.getenv("API_KEY")
)

EMBED_MODEL=os.getenv("EMBEDDING_MODEL")
KB_FOLDER=os.getenv("KB_FOLDER")
VECTOR_STORE=os.getenv("VECTOR_STORE")

#Get the embedding of a text from Embedding Model
def get_embedding(text):
  response = client.embeddings.create(
    model=EMBED_MODEL,
    input=text
  )
  return response.data[0].embedding

# Get the Cosine Similarity between two vectors
def cosine_similarity(A, B):
  dot_prod = numpy.dot(A, B)
  norm_a = numpy.linalg.norm(A)
  norm_b = numpy.linalg.norm(B)

  return dot_prod/(norm_a*norm_b)


#Load the documents
def load_documents(folder=KB_FOLDER):
  docments = []

  for filename in sorted(os.listdir(folder)):
    if not filename.endswith(".txt"): continue

    path = os.path.join(folder, filename)
    with open(path, "r", encoding="utf-8") as f:
      docments.append(
        {
          "source": filename,
          "text": f.read()
        }
      )
  return docments

#Add embeddings in the chunks
def add_embeddings(chunks):
  for chunk in chunks:
    vector = get_embedding(chunk["text"])
    chunk["vector"] = vector
  
  return chunks

#Search method to search the top_k chunks for a user_query
def search(query, chunks, top_k):
  query_vector = get_embedding(query)

  results = []
  for chunk in chunks:
    score = cosine_similarity(chunk["vector"], query_vector)
    results.append(
      {
        "score": float(score),
        "source": chunk["source"],
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "size": len(chunk["text"])
      }
    )
  
  results.sort(key=lambda rec: rec["score"], reverse=True)
  return results[:top_k]

#testing method
def test(user_query, chunks, top_k=3):
  print(f"\n User Query: {user_query}\n")
  results = search(user_query, chunks, top_k)

  for r in results:
    print(f" {r['score']:.2f} - {r['source']} - {r['size']} characters")
    print(f"    {r['text']}\n")



#Takes one document and create multiple chunks from it. One section = One chunk
def chunk_document(document):
  chunks = []

  chunk_id = 0
  for part in document["text"].split("\n\n"):
    part = part.strip()
    if part:
      chunks.append(
        {
          "source": document["source"],
          "chunk_id": chunk_id,
          "text": part
        }
      )
    chunk_id += 1
  return chunks 

def create_chunks():
  documents = load_documents();
  chunks = []

  for document in documents:
    chunks.extend(chunk_document(document))
  return chunks

def save_vector_store(chunks):
  with open(VECTOR_STORE, 'w', encoding="utf-8") as f:
    json.dump(chunks, f, indent=4)
  
def load_vector_store():
  with open(VECTOR_STORE, 'r', encoding="utf-8") as f:
    return json.load(f)

#Lets start testing
chunks = []

if os.path.exists(VECTOR_STORE):
  chunks = load_vector_store()
  print(f"Loaded {len(chunks)} chunks from the vector store file {VECTOR_STORE}")
else:
  chunks = create_chunks()
  chunks = add_embeddings(chunks)
  save_vector_store(chunks)
  print(f"Created and embedded {len(chunks)} chunks and saved to the vector store file {VECTOR_STORE}.")

print(f"We have embedded {len(chunks)} chunks from our knowledge base.")

user_query = "What are the benefits of Maternity Leaves?"
test(user_query, chunks, 3)