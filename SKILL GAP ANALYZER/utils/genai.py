from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

def get_course_recommendations(reviews_text, user_specific_requirements):
    """
    Reads the reviews from reviews_file, builds the prompt with the user-specific requirements,
    sends it to the Gemini model, and returns the recommendations as text.
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    # Set up the Gemini model
    model = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.3, api_key=api_key)
    
    # Define the prompt template
    prompt_template = PromptTemplate.from_template(
        """You are an AI assistant that helps users choose the best online courses which align with the user's specific requirements. 
I will provide you with a list of semantically relevant course reviews and the user requirements.
Your task is to analyze the feedback and recommend the top 3 courses based on learner satisfaction, course quality, and usefulness in accordance with user requirements.

### Course Reviews:
{reviews}

### User requirements:
{user_specific_requirements}

### Instructions:
- Consider aspects like why you choose a review and how it aligns with the user_specific_requirements.
- Rank the top 3 courses and explain why they are the best choices.
- Output the recommendations in the following format:

Recommended Courses:
1. Course Name: 
   (Course Name)
   Course Url: 
   (Course Url)
   Reason for selection: 
   (Reason of Selection)
2. Course Name: 
   (Course Name)
   Course Url: 
   (Course Url)
   Reason for selection: 
   (Reason of Selection)
3. Course Name: 
   (Course Name)
   Course Url: 
   (Course Url)
   Reason for selection: 
   (Reason of Selection)



Make sure your response is concise, informative, and user-friendly."""
    )
    
    # Create the chain by combining the prompt template and the model.
    chain = prompt_template | model
    
    # Read the reviews from file

    
    # Prepare the input dictionary for the chain
    inputs = {
        "reviews": reviews_text,
        "user_specific_requirements": user_specific_requirements
    }
    
    # Invoke the chain and return the response content.
    response = chain.invoke(inputs)
    return response.content

if __name__ == "__main__":
    # Specify the input reviews file and the user-specific requirements.
    reviews_file = "temp.txt"
    user_specific_requirements = "real life projects"
    
    recommendations = get_course_recommendations(reviews_file, user_specific_requirements)
    print("Recommendations:\n")
    print(recommendations)
