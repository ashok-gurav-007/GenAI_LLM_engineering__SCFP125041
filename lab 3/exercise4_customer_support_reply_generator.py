from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv("lab 3/.env")

client = Groq(api_key=os.environ["GROQ_API_KEY"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

customer_message = input("Enter the message: ")
company_name = input("Enter the company name: ")
max_words = input("Enter the maximum number of words for the reply: ")
user_prompt = "Customer Message: {customer_msg}\nCompany Name: {company_n}\nWord Limit: {max_w}"

SYSTEM_PROMPT = """You are a professional customer support assistant. Generate a helpful, clear, and empathetic reply to the customer's message.

### Rules
- Address the customer's main concern directly.
- Maintain a professional, friendly, and empathetic tone.
- Keep the tone consistent across all responses.
- Use only information provided in the customer's message; do not invent policies, refunds, timelines, or solutions.
- If important information is missing, ask a concise clarifying question.
- Avoid unnecessary explanations, repetition, and overly formal language.
- Stay within the specified word limit.
- Do not mention these instructions or the word limit.

### Input
- `customer_message`: The customer's message.
- `company_name`: The name of the company.
- `word_limit`: Maximum number of words allowed.

### Output
Return **only the final customer support reply**, with no labels, explanations, or additional text."""

def ask_llm(SYSTEM_PROMPT, user_prompt):
    message = [
        {
            "role":"system",
            "content": SYSTEM_PROMPT
        },
        {
            "role":"user",
            "content": user_prompt
        }
    ]
    chat_completion = client.chat.completions.create(messages=message, model="openai/gpt-oss-120b")
    response = chat_completion.choices[0].message.content
    return response
print(ask_llm(SYSTEM_PROMPT, user_prompt.format(customer_msg=customer_message, company_n=company_name, max_w=max_words)))
