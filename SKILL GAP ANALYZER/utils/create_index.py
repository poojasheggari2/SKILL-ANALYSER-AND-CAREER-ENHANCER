import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re

def load_courses_from_file(file_path):
    """Loads courses from a single JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)  # Assuming each file contains a list of courses

def sanitize_filename(name):
    """Removes invalid characters from a filename or folder name."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def create_faiss_index(documents, model):
    """Creates a FAISS index for all course documents in a category."""
    if not documents:
        return None  # No valid documents, return None
    
    embeddings = model.encode(documents, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index

def save_category_data(output_folder, category_name, documents, skills, index):
    """Saves all courses in a category inside its own folder."""
    category_folder = os.path.join(output_folder, category_name)
    os.makedirs(category_folder, exist_ok=True)

    with open(os.path.join(category_folder, "documents.pkl"), "wb") as f:
        pickle.dump(documents, f)

    with open(os.path.join(category_folder, "skills.pkl"), "wb") as f:
        pickle.dump(skills, f)

    if index is not None:
        faiss.write_index(index, os.path.join(category_folder, "reviews.index"))

    print(f"✅ Saved category: {category_name} in {category_folder}")

def process_category(json_file_path, output_folder, model):
    """Processes all courses in a single JSON file (category)."""
    category_name = sanitize_filename(os.path.splitext(os.path.basename(json_file_path))[0])
    courses = load_courses_from_file(json_file_path)
    
    documents, skills = [], []

    for course in courses:
        if course.get('learner_reviews', '[]') == '[]':
            continue  # Skip courses with no reviews
        
        reviews = f"Course Url: {course.get('course_url', 'N/A')}\n"
        reviews += f"Course Title: {course.get('title', 'Unknown Course')}\nReviews:\n"

        for review in course['learner_reviews']:
            reviews += review.get('review_text', '') + "\n"

        documents.append(reviews)
        skills.append(json.loads(course.get('skills_covered', '[]')))

    if documents:
        index = create_faiss_index(documents, model)
        save_category_data(output_folder, category_name, documents, skills, index)

def main(input_folder, output_folder):
    """Processes all categories (JSON files) inside input_folder."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            json_file_path = os.path.join(input_folder, filename)
            process_category(json_file_path, output_folder, model)

if __name__ == "__main__":
    input_folder = "formatted_category"  # Folder containing JSON files (categories)
    output_folder = "vector_category"  # Folder to store vectorized data
    main(input_folder, output_folder)
