import os
from groq import Groq
from dotenv import load_dotenv # <-- Add this import

# --- Add this line to load variables from .env ---
load_dotenv() 

def get_groq_answer(query, context):
    # Get API key from environment variable (now loaded from .env)
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        print("Error: GROQ_API_KEY environment variable not set. "
              "Please ensure it's in your .env file or set as an OS environment variable.")
        return "An error occurred: API key not configured."

    client = Groq(api_key=groq_api_key) 

    model_name = "llama-3.3-70b-versatile" 

    prompt = f"""You are a helpful and professional assistant specializing in supply chain management. Your goal is to provide clear, concise, and easy-to-understand answers to clients, avoiding technical jargon.

    Please answer the following question based *only* on the provided context. Do not mention the source of the information or refer to the context itself. Simply provide the answer in a polite, normal, and client-friendly way.

    If the provided context does not contain enough information to answer the question, please politely respond with: "I don't have enough information to answer that at the moment. Please provide more details if possible."

    Question: {query}

    Context:
    {context}

    Answer:"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
            temperature=0.1, 
            max_tokens=500, 
            top_p=1,
            stop=None,
            stream=False,
        )
        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error during Groq API call: {e}")
        return f"An error occurred while communicating with Groq: {e}"
# # --- Sample Usage ---
# if __name__ == '__main__':
#     print("--- Running Sample Groq API Call ---")

#     # Example query and context
#     test_query = "What is the lead time for Widget A?"
#     test_context = """
#     Node: PROD01, Attributes: ProductID: PROD01, Name: Widget A, Description: Standard Widget, Material: Steel
#     Relationship: PROD01 --[supplied_by]--> SUP001, Neighbor Attributes: SupplierID: SUP001, Name: Alpha Components, Location: China, LeadTime: 10 days
#     Node: PROD02, Attributes: ProductID: PROD02, Name: Gadget B, Description: Advanced Gadget, Material: Plastic
#     Relationship: PROD02 --[supplied_by]--> SUP002, Neighbor Attributes: SupplierID: SUP002, Name: Beta Plastics, Location: USA, LeadTime: 7 days
#     """
    
#     print(f"Test Query: {test_query}")
#     print(f"Provided Context:\n{test_context}")

#     answer = get_groq_answer(test_query, test_context)
    
#     print("\n--- Groq's Answer ---")
#     print(answer)
#     print("\n----------------------")