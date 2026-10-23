
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=api_key
)


class ResearchData(BaseModel):
    research_question: str = Field(
        description="The main research question addressed by the paper"
    )
    method: str = Field(
        description="The research method or approach used in the paper"
    )
    key_finding: str = Field(
        description="The main finding or result of the research"
    )


parser = PydanticOutputParser(pydantic_object=ResearchData)

step1_prompt = PromptTemplate(
    template="""Analyze the following research paper abstract.

Extract these three details:
1. Research question
2. Research method
3. Key finding

Base your answer only on the information in the abstract.
Do not invent missing details.

Research Paper Abstract:
{abstract_text}

{format_instructions}
""",
    input_variables=["abstract_text"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

step1_chain = step1_prompt | llm | parser

step2_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a science communicator who explains research
findings to a general audience.

Write a clear, concise summary in simple language.
Use only the research question, method, and key finding provided.
Do not add unsupported information."""
        ),
        (
            "human",
            """Research Question: {research_question}

Research Method: {method}

Key Finding: {key_finding}

Write a plain-language summary that a non-expert can understand."""
        )
    ]
)

step2_parser = StrOutputParser()

step2_chain = step2_prompt | llm | step2_parser

abstract_text = input("Enter the research paper abstract: ")

result = step1_chain.invoke(
    {"abstract_text": abstract_text}
)

print("\nStructured Research Extraction:")
print(result.model_dump_json(indent=2))

layperson_summary = step2_chain.invoke(
    {
        "research_question": result.research_question,
        "method": result.method,
        "key_finding": result.key_finding
    }
)

print("\nPlain-Language Research Summary:")
print(layperson_summary)
