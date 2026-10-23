
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


class CustomerComplaint(BaseModel):
    complaint: str = Field(description="Main issue reported by the customer")
    product_or_feature: str = Field(description="Product or feature related to the issue")
    sentiment: str = Field(description="Emotion or attitude expressed in the review")


structured_parser = PydanticOutputParser(
    pydantic_object=CustomerComplaint
)

customer_review = input("Enter the customer's review: ")

analysis_prompt = PromptTemplate(
    template="""Read the customer review and identify the following:
1. The main complaint
2. The product or feature involved
3. The customer's sentiment

Review:
{review}

Return the extracted information in the required JSON format:
{format_instructions}
""",
    input_variables=["review"],
    partial_variables={
        "format_instructions": structured_parser.get_format_instructions()
    }
)

analysis_chain = analysis_prompt | llm | structured_parser

complaint_info = analysis_chain.invoke(
    {"review": customer_review}
)

print("\nExtracted Complaint Information:")
print(complaint_info.model_dump_json(indent=2))


ticket_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You prepare clear and concise support tickets for an internal customer service team."
        ),
        (
            "human",
            """Write a support ticket summary based on these details.

Main Complaint: {complaint}
Related Product or Feature: {product_or_feature}
Customer Sentiment: {sentiment}

Include the main issue and the customer's sentiment in a brief, professional summary."""
        )
    ]
)

text_parser = StrOutputParser()

ticket_chain = ticket_prompt | llm | text_parser

summary = ticket_chain.invoke(
    {
        "complaint": complaint_info.complaint,
        "product_or_feature": complaint_info.product_or_feature,
        "sentiment": complaint_info.sentiment
    }
)

print("\nGenerated Support Ticket Summary:")
print(summary)

