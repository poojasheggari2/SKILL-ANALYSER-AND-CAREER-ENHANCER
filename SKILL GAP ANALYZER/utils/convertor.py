import json

def split_reviews_keep_fields(input_filename, output_filename):
    # Load the JSON file
    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []

    for course in data:
        course_copy = course.copy()  # Keep all fields
        learner_reviews = course_copy.get("learner_reviews", "[]")  # Get learner_reviews safely

        try:
            reviews = json.loads(learner_reviews)  # Convert string to list
        except json.JSONDecodeError:
            reviews = []  # If invalid JSON, set as empty list

        if isinstance(reviews, list) and reviews:
            total_reviews = len(reviews)
            chunks = [reviews[i:i + 20] for i in range(0, total_reviews, 20)]

            for chunk in chunks:
                new_course = course.copy()  # Keep all fields unchanged
                new_course["learner_reviews"] = chunk  # Replace learner_reviews with a chunk
                processed_data.append(new_course)
        else:
            course_copy["learner_reviews"] = "[]"  # Keep empty list as string if no reviews
            processed_data.append(course_copy)

    # Save the modified data
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

# Example usage
split_reviews_keep_fields("temp.json", "output.json")
