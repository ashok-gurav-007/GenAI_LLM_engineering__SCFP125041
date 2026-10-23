from dotenv import load_dotenv
from groq import Groq
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def ask_llm(system_prompt, user_prompt):
    message = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    chat_completion = client.chat.completions.create(messages=message,model="openai/gpt-oss-120b")
    response = chat_completion.choices[0].message.content
    return response

draft_text = input("Enter draft text: ")
issue_to_fix = input("Enter issue to fix: ")


SYSTEM_PROMPT = """
You are a professional content editor.

Your task is to revise the given draft to fix one specific issue.

Follow these rules:
1. Fix only the specified issue.
2. Preserve the original meaning and important information.
3. Do not introduce unrelated changes.
4. Keep the writing natural and clear.
5. Return only the revised draft.
"""


user_prompt = """
Draft:
{draft_text}

Issue to fix:
{issue_to_fix}

Revise the draft to fix the specified issue.
"""

round1 = ask_llm(SYSTEM_PROMPT,user_prompt.format(draft_text=draft_text,issue_to_fix=issue_to_fix))
print("\nRound 1:")
print(round1)


round2 = ask_llm(SYSTEM_PROMPT,user_prompt.format(draft_text=round1,issue_to_fix=issue_to_fix))
print("\nRound 2:")
print(round2)


round3 = ask_llm(SYSTEM_PROMPT,user_prompt.format(draft_text=round2,issue_to_fix=issue_to_fix))
print("\nRound 3:")
print(round3)


print("\nFinal Revised Version:")
print(round3)