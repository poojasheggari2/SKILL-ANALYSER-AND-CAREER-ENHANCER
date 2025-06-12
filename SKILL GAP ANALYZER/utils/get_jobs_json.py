import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model path (change this if needed)
MODEL_PATH = "C:/Users/harsh/Desktop/SkillSync_AI/models/TinyLlama-1.1B-Chat-v1.0"

def initialize_model():
    """Loads the TinyLlama model and tokenizer locally."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

def get_llm_pipeline(tokenizer, model):
    """Returns a LangChain-compatible pipeline."""
    from transformers import pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

def get_output_parser():
    """Defines the structured JSON output format."""
    response_schemas = [
        ResponseSchema(name="category", description="Category from the predefined list."),
        ResponseSchema(name="skills", description="List of required skills or 'None'.", type="list"),
        ResponseSchema(name="experience", description="Experience range as [low, high] or 'None'.", type="list")
    ]
    return StructuredOutputParser.from_response_schemas(response_schemas)

def create_prompt(output_parser):
    """Creates and returns the LangChain prompt template."""
    format_instructions = output_parser.get_format_instructions()
    example_output = '''{
        "category": "Software Engineering",
        "skills": ["Python", "Machine Learning"],
        "experience": [2, 5]
    }'''
    return PromptTemplate(
        input_variables=["job_description"],
        template="""
        You are an AI that extracts structured job information.
        Given the following job description:
        {job_description}
        
        Please provide a JSON output in this format:
        {format_instructions}
        
        Example output:
        {example_output}
        """,
        partial_variables={"format_instructions": format_instructions, "example_output": example_output}
    )

def validate_json(output):
    """Validates JSON output against the schema."""
    try:
        json_obj = json.loads(output)
        if all(k in json_obj for k in ["category", "skills", "experience"]):
            return json_obj
    except Exception as e:
        print("Invalid JSON output:", e)
    return None

def extract_job_info(job_description, llm, prompt_template, output_parser):
    """Runs the LLM and ensures JSON output."""
    chain = prompt_template | llm | RunnablePassthrough()
    raw_output = chain.invoke({"job_description": job_description})
    structured_output = output_parser.parse(raw_output)
    return validate_json(structured_output) or {}

if __name__ == "__main__":
    tokenizer, model = initialize_model()
    llm = get_llm_pipeline(tokenizer, model)
    output_parser = get_output_parser()
    prompt_template = create_prompt(output_parser)
    
    sample_job_description = "We are looking for a Python Developer with experience in Flask and APIs. Minimum 2 years of experience required."
    structured_data = extract_job_info(sample_job_description, llm, prompt_template, output_parser)
    
    print(json.dumps(structured_data, indent=2))
