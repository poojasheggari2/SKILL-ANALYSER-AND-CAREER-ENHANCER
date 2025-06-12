import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_data(input_folder):
    """
    Load documents, skills, and the FAISS index from the input folder.
    """
    documents_path = os.path.join(input_folder, "documents.pkl")
    skills_path = os.path.join(input_folder, "skills.pkl")
    index_path = os.path.join(input_folder, "reviews.index")
    
    with open(documents_path, "rb") as f:
        documents = pickle.load(f)
        
    with open(skills_path, "rb") as f:
        skills = pickle.load(f)
        
    index = faiss.read_index(index_path)
    return documents, skills, index

def filter_indices_by_skills(skills, required_skills):
    """
    Return a list of indices where any required skill is present.
    Each element in `skills` is assumed to be a list of strings.
    """
    allowed_idx = []
    for i, skill_list in enumerate(skills):
        for req in required_skills:
            if req in skill_list:
                allowed_idx.append(i)
                break  # Once found, move to next document
    return allowed_idx

def create_sub_index(index, allowed_idx):
    """
    Reconstruct vectors from the original index for allowed indices,
    and create a new FAISS index (sub-index) from these vectors.
    
    Returns:
        sub_index: The new FAISS index for filtered vectors.
        sub_index_to_original: Mapping from sub-index to original index.
    """
    filtered_vectors = np.array([index.reconstruct(i) for i in allowed_idx if i < index.ntotal])
    
    if filtered_vectors.size == 0:
        print("No matching vectors found for the specified skills.")
        return None, None
    
    dimension = filtered_vectors.shape[1]
    sub_index = faiss.IndexFlatL2(dimension)
    sub_index.add(filtered_vectors)
    
    # Create a mapping from sub-index positions to original indices
    sub_index_to_original = {i: allowed_idx[i] for i in range(len(allowed_idx))}
    return sub_index, sub_index_to_original

def search_similar_reviews(query, model, sub_index, sub_index_to_original, documents, top_k=10):
    """
    Encode the query, search the sub-index, and return matching documents.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = sub_index.search(query_embedding, top_k)
    
    results = []
    for idx in indices[0]:
        original_idx = sub_index_to_original.get(idx)
        if original_idx is not None:
            results.append(documents[original_idx])
    return results

def process_(input_folder, required_skills, query):
    """
    Process the input data and return the filtered results as a text string.
    """
    # Load data from the input folder
    documents, skills, index = load_data(input_folder)
    
    # Filter indices based on required skills
    allowed_idx = filter_indices_by_skills(skills, required_skills)
    
    # Create a sub-index from the allowed indices
    sub_index, sub_index_to_original = create_sub_index(index, allowed_idx)
    if sub_index is None:
        return "No matching vectors found for the specified skills."
    
    # Initialize the sentence transformer model once
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Perform the similarity search using the query
    results = search_similar_reviews(query, model, sub_index, sub_index_to_original, documents)
    
    # Return the results as a single text string (each document on a new line)
    return "\n".join(results)

if __name__ == "__main__":
    # Update these paths as needed
    input_folder = ""    # Folder containing documents.pkl, skills.pkl, and reviews.index
    
    # Define the required skills and the search query
    required_skills = ['SQL', 'Linux', 'Deception Techniques']
    query = "basics covered"
    
    # Process the data and get results as text
    output_text = process_(input_folder, required_skills, query)
    print("Filtered search results:\n")
    print(output_text)

