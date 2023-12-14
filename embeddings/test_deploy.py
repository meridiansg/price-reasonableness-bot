import os
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

from create_embeddings import model, createEmbeddings
from extract_keywords import extractKeywords

model_path = os.getenv('BGE_MODEL_PATH')
print("BGE Model Path: ", model_path)

print("Loading BGE model...")
bge_model = model(model_path)[0]
print("BGE model loaded.")

# Define the path to the embeddings file
embedded_path = os.getenv('BGE_COMBINED_EMBEDDED_PATH')

# Check if embeddings have already been created
if os.path.exists(embedded_path):
    print("Embeddings have already been created.")
else:
    print("Embeddings have not been created yet, creating embeddings...")
    createEmbeddings(model_path)

medical_queries = [
    "What are the common symptoms and treatments for Type 2 Diabetes?",
    "How does chemotherapy work in treating cancer?",
    "Can you provide information on the causes and management of Chronic Obstructive Pulmonary Disease?",
    "How is Rheumatoid Arthritis diagnosed, and what are the available treatments?",
    "Can you explain the progression and stages of Alzheimer's disease, and what medications are typically used for management?"
]

for query in medical_queries:
    extractKeywords(embedded_path=embedded_path, bge_model=bge_model, model_path=model_path, query=query, top_k=5)
