from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv("lab 3/.env")

client = Groq(api_key=os.environ["GROQ_API_KEY"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key= os.environ["GROQ_API_KEY"])

def ask_llm(system_prompt, user_prompt):
    message = [
        {
            "role":"system",
            "content": system_prompt
        },
        {
            "role":"user",
            "content":user_prompt
        }
    ]
    chat_completion = client.chat.completions.create(messages = message, model="openai/gpt-oss-120b")
    response = chat_completion.choices[0].message.content
    return response

patient_name = input("Enter patient name: ")
patient_notes = input("Enter your notes: ")

system_prompt="You are a doctor and you will get patient's raw constulation notes and generate a clean doctor's summary. The Summary with fixed sections are Symptopms, Diagnosis, Recommendation - in the same format eery time, regardless of input."
user_prompt = "My Name is {name} and my notes are {notes}"

print(ask_llm(system_prompt,user_prompt.format(name = patient_name,notes = patient_notes)))