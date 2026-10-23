
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser
)
from pydantic import BaseModel, Field

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)


class SupportSummary(BaseModel):
    issue: str = Field(
        description="Main issue reported by the customer"
    )
    customer_tone: str = Field(
        description="Customer's tone"
    )
    resolution_attempted: str = Field(
        description="Resolution steps already attempted"
    )


step1_parser = PydanticOutputParser(
    pydantic_object=SupportSummary
)


step1_prompt = PromptTemplate(
    template="""
Analyze the following customer support chat log.

Extract and normalize:
- The main issue
- The customer's tone
- The resolution attempted so far

Do not invent information that is not present in the chat log.
If a resolution has not been attempted, state "Not specified".

Chat Log:
{chat_log_text}

{format_instructions}
""",
    input_variables=["chat_log_text"],
    partial_variables={
        "format_instructions": step1_parser.get_format_instructions()
    }
)


step1_chain = step1_prompt | llm | step1_parser

class EscalationCheck(BaseModel):
    missing_fields: list[str] = Field(
        description="Information missing for escalation"
    )
    is_ready_to_escalate: bool = Field(
        description="Whether the issue is ready for escalation"
    )


step2_parser = PydanticOutputParser(
    pydantic_object=EscalationCheck
)


step2_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an escalation-readiness reviewer.

Check the structured support summary for information
needed to escalate the issue.

Important information may include:
- Account ID
- Error code, if applicable
- Specific issue details
- Resolution steps already attempted

Rules:
- Identify important information that is missing.
- Add missing information to missing_fields.
- Set is_ready_to_escalate to false if essential information
  is missing.
- Set is_ready_to_escalate to true if sufficient information
  is available to proceed.
- Do not assume that missing information exists.
- Use only the structured summary provided.

{format_instructions}
"""
    ),
    (
        "user",
        """
Structured Support Summary:

Issue: {issue}
Customer Tone: {customer_tone}
Resolution Attempted: {resolution_attempted}

Check whether this issue is ready for escalation.
"""
    )
])


step2_prompt = step2_prompt.partial(
    format_instructions=step2_parser.get_format_instructions()
)

step2_chain = step2_prompt | llm | step2_parser


step3_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a support operations lead.

Create a prioritized escalation report based only on the
structured support summary and escalation-readiness check.

Include:
- Priority: High, Medium, or Low
- Issue summary
- Customer tone
- Resolution attempted
- Missing information
- Escalation readiness
- Recommended next action

Guidelines:
- Explain the priority based on the information provided.
- Do not invent account IDs, error codes, or troubleshooting
  steps.
- If essential information is missing, recommend collecting
  it before escalation.
- If the issue is ready for escalation, recommend the
  appropriate next step.
- Keep the report clear and concise.
"""
    ),
    (
        "user",
        """
Structured Support Summary:

Issue: {issue}
Customer Tone: {customer_tone}
Resolution Attempted: {resolution_attempted}

Escalation Check:

Missing Fields: {missing_fields}
Ready to Escalate: {is_ready_to_escalate}

Generate the escalation report.
"""
    )
])


step3_parser = StrOutputParser()

step3_chain = step3_prompt | llm | step3_parser

print("\n" + "=" * 50)
print("CUSTOMER SUPPORT ESCALATION PIPELINE")
print("=" * 50)

print("\nEnter the customer support chat log.")
print("Type END on a separate line when finished:\n")

chat_log_lines = []

while True:
    line = input()

    if line.strip().upper() == "END":
        break

    chat_log_lines.append(line)


chat_log_text = "\n".join(chat_log_lines).strip()

if not chat_log_text:
    raise ValueError("Customer support chat log cannot be empty.")



print("\nSTEP 1: Extracting Structured Support Summary...")

step1_result = step1_chain.invoke({
    "chat_log_text": chat_log_text
})

print("\nSTEP 1: Structured Summary")
print(step1_result.model_dump_json(indent=2))


print("\nSTEP 2: Checking Escalation Readiness...")

step2_result = step2_chain.invoke({
    "issue": step1_result.issue,
    "customer_tone": step1_result.customer_tone,
    "resolution_attempted": step1_result.resolution_attempted
})

print("\nSTEP 2: Escalation Readiness")
print(step2_result.model_dump_json(indent=2))


print("\nSTEP 3: Generating Escalation Report...")

step3_result = step3_chain.invoke({
    "issue": step1_result.issue,
    "customer_tone": step1_result.customer_tone,
    "resolution_attempted": step1_result.resolution_attempted,
    "missing_fields": step2_result.missing_fields,
    "is_ready_to_escalate": step2_result.is_ready_to_escalate
})

print("\nSTEP 3: Escalation Report")
print(step3_result)

print("\n" + "=" * 50)
print("Escalation analysis completed successfully.")
print("=" * 50)
