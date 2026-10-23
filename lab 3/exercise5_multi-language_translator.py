from dotenv import load_dotenv
from groq import Groq
import os


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def ask_llm(system_prompt,user_prompt):
    message = [
        {
            "role":"system",
            "content":system_prompt
        },
        {
            "role":"user",
            "content":user_prompt
        }
    ]

    chat_completion = client.chat.completions.create(messages=message,model="openai/gpt-oss-120b")
    response = chat_completion.choices[0].message.content
    return response

sentence = input("Enter sentence: ")
target_language = input("Enter target language: ")
formality = input("Enter Formality: ")

SYSTEM_PROMPT = """You are a professional translation assistant.

Your task is to translate the user's sentence into the specified target language while preserving its original meaning, intent, and context.

Follow these rules:
1. Translate the sentence accurately and naturally.
2. Use the requested target language.
3. Match the requested formality level:
   - **Formal**: professional, respectful, and appropriate for official or workplace communication.
   - **Neutral**: natural and standard language suitable for general communication.
   - **Informal**: casual, friendly, and conversational language.
4. Do not add explanations, opinions, or information that is not present in the original sentence.
5. Preserve names, numbers, dates, technical terms, and important details unless they need to be naturally adapted to the target language.
6. If the original sentence is ambiguous, preserve the ambiguity rather than inventing meaning.
7. Return only the translated sentence.

Input:
- Source sentence: {sentence}
- Target language: {target_language}
- Formality level: {formality}

**Output:** Only the translated sentence, written naturally in the target language and matching the requested formality level."""

user_prompt = """Sentence:{sentence} Target Language:{target_language} Formality level:{formality}"""

print(ask_llm(SYSTEM_PROMPT,user_prompt.format(sentence=sentence,target_language=target_language,formality=formality)))