import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from create_embeddings import createEmbeddings, model

# Load Environment Variables
load_dotenv()

# Function to extract keywords
def extractKeywords(embedded_path, bge_model, model_path,query: str, top_k: int = 5):
    query = [query]


    # Load the embeddings

    embeddings_df = pd.read_json(embedded_path)  # Load DataFrame and store in dictionary
    # Display the loaded DataFrame

    embeddings_matrix = np.vstack(embeddings_df['embedding'].apply(np.array))

    # Call check_normalisation for embeddings_matrix

    # if not check_normalisation(embeddings_matrix):
    #     print("\033[91mWarning: The embeddings matrix is not normalized!\033[0m")

    no_queries = len(query)
    top_k = top_k
    query_list = []

    query_embeddings = bge_model.encode_queries(query)
    # Call check_normalisation for query_embeddings
    # check_normalisation(query_embeddings)

    embeddings_df = embeddings_df.reset_index()

    print('Queries received. Please wait a moment while we process your queries.')
    output_dir = os.getenv('OUTPUT_DIR')
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    output_filepath = os.path.join(output_dir, f"{timestamp_str}.json")

    metadata = {
        "timestamp": timestamp_str,
        "dataset": 'combined',
        "no_queries": no_queries,
        "top_k": top_k,
        "method": "bge",
        "queries": [],
    }

    for i, query in enumerate(query):
        scores = query_embeddings[i] @ embeddings_matrix.T
        scores_df = pd.DataFrame(scores, columns=['Similarity']).sort_values(by='Similarity',
                                                                             ascending=False).head(top_k)
        scores_df = scores_df.merge(embeddings_df, left_index=True, right_index=True) \
            .rename(columns={'index': 'Index Location'}) \
            .drop(columns=['for_embedding', 'relationship', 'embedding'])

        response_metadata = {
            "query": query,
            "response": scores_df.to_dict(orient='records')
        }
        metadata["queries"].append(response_metadata)

    with open(output_filepath, 'w') as output_file:
        json.dump(metadata, output_file, indent=4)

    for response_metadata in metadata["queries"]:
        query = response_metadata["query"]
        response = pd.DataFrame(response_metadata["response"])
        print(f'\nTop {top_k} results for query: {query}')


    print(f"Query: {query}\nResults and metadata saved in '{output_filepath}'")
    return metadata



