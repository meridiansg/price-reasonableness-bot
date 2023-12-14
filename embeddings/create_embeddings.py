import os

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

def model(bge_model_path):
    from FlagEmbedding import FlagModel

    bge_model = FlagModel(bge_model_path,
                          query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                          normalize_embeddings=True,
                          use_fp16=True)
    return bge_model, bge_model_path


def createEmbeddings(model_path):
    bge_model = model(model_path)[0]
    dataset_list = ['combined']

    # CSV Paths
    csv_paths = {
        'combined': os.getenv('COMBINED_DATASET_PATH')
    }

    # Embedded Paths for bge model
    bge_embedded_paths = {
        'combined': os.getenv('BGE_COMBINED_EMBEDDED_PATH'),
    }

    # Model Path for bge model
    bge_model_path = os.getenv('BGE_MODEL_PATH')

    # Re-embed dataset using BGE (FlagEmbedding)
    bge_embedding = True

    if bge_embedding:
        bge_online = True

    # Dictionary to hold the DataFrames
    dfs = {}

    for dataset in dataset_list:
        data = pd.read_csv(csv_paths[dataset])

        # Handle different datasets with specific requirements
        if dataset == 'lailo':
            df = pd.DataFrame(data).drop(
                columns=['New LAILO code', 'LAILO_Code', 'TD_Codes', 'MILO_Codes', 'Assess_Type_Codes',
                         'Module sequence']
            )
        elif dataset == 'combined':
            df = pd.DataFrame(data).drop(
                columns=['LAILO_Code', 'TD_Codes', 'MILO_Codes', 'Assess_Type_Codes', 'Module sequence', 'MILO_Year',
                         'Sequence', 'Module_Seq', 'level 2 code', 'level 3 -TD Code', 'Module_Sheet_Year',
                         'Module_Sheet_Sequence']
            )
        else:
            df = pd.DataFrame(data)

        # Common operations for all datasets
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')].dropna()
        df = df.reset_index(drop=True)
        # Store the DataFrame in the dfs dictionary with the dataset name as the key
        dfs[dataset] = df

    # Define datasets and their operations
    datasets_operations = {
        'combined': {
            'relationship': lambda
                row: f"In the '{row['Module']}' module of the '{row['Longitudinal_Course']}' longitudinal course, a {row['LA_Type']} session focuses on '{row['LA_Topic']}'. The learning objective is: '{row['LAILO']}', and is aligned with MILO code {row['MILO_Code']}, focusing on '{row['MILO_Type_Name']}', and specifically, '{row['MILO']}'. This belongs to the theme of '{row['level 1 - Theme']}' under the fundamental of '{row['level 2 - Fundamental']}' in the tertiary domain of '{row['level 3 - Tertiary Domain (TD)']}.",

            'for_embedding': lambda
                row: f"Year {row['LAILO_Year']}, {row['Module']}, {row['Longitudinal_Course']}, {row['LA_Type']}, {row['LA_Topic']}, {row['New LAILO code']}, {row['LAILO']}, {row['MILO_Type']}, {row['MILO_Type_Name']}, {row['MILO_Code']}, {row['MILO']}, {row['level 1 - Theme']}, {row['level 2 - Fundamental']}, {row['level 3 - Tertiary Domain (TD)']}"
        }
    }

    # Initialize a dictionary to hold the combined DataFrames
    combined_dfs = {}

    # Loop through datasets_operations and apply the functions dynamically
    for dataset, operations in datasets_operations.items():
        # Access the original DataFrame from the dfs dictionary
        combined_df = dfs[dataset].copy()

        # Apply the lambda functions
        for column, func in operations.items():
            combined_df[column] = combined_df.apply(func, axis=1)

        # Store the combined DataFrame in the combined_dfs dictionary
        combined_dfs[dataset] = combined_df

    def check_normalisation(matrix, tolerance=1e-9):
        norms = np.linalg.norm(matrix, axis=1)
        is_normalized = np.all(np.abs(norms - 1.0) < tolerance)
        return is_normalized



    def BGE_Create_Embedding():
        for dataset in dataset_list:
            bge_copied_df = combined_dfs[dataset].copy()  # Access the DataFrame from combined_dfs dictionary

            bge_embedded_path = bge_embedded_paths[dataset]  # Get the corresponding embedded path

            passage_embeddings = bge_model.encode(list(bge_copied_df['for_embedding'])).tolist()
            # Convert numpy array to list and merge with the original dataframe
            passage_embeddings_dataframe = pd.DataFrame(passage_embeddings)

            passage_embeddings_dataframe = pd.DataFrame(
                passage_embeddings_dataframe.apply(lambda row: row.tolist(), axis=1), columns=['embedding'])
            # Join the DataFrames based on index.
            bge_embeddings = bge_copied_df.join(passage_embeddings_dataframe)

            # Save to JSON and Display
            bge_embeddings.to_json(bge_embedded_path)

    if bge_embedding:
        BGE_Create_Embedding()
        print('BGE embeddings have been created.')
    else:
        print("BGE embeddings have already been created.")
