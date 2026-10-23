
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# ==========================================
# INITIALIZATION
# ==========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. Please check your .env file."
    )

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=api_key
)


# ==========================================
# STEP 1: EXTRACT INTERVIEW QUESTIONS AND ANSWERS
# ==========================================

class QuestionAnswer(BaseModel):
    question: str = Field(
        description="The interview question asked"
    )
    answer: str = Field(
        description="The candidate's answer to the question"
    )


class InterviewAnswers(BaseModel):
    answers: list[QuestionAnswer] = Field(
        description="List of interview questions and candidate answers"
    )


step1_parser = PydanticOutputParser(
    pydantic_object=InterviewAnswers
)

step1_prompt = PromptTemplate(
    template="""Analyze the interview transcript below.

Extract every interview question and its corresponding candidate answer.

Instructions:
- Match each question with the appropriate candidate answer.
- Preserve the meaning of the original answers.
- Do not invent questions or answers.
- If no question-answer pairs are present, return an empty list.

Interview Transcript:
{transcript_text}

{format_instructions}
""",
    input_variables=["transcript_text"],
    partial_variables={
        "format_instructions": step1_parser.get_format_instructions()
    }
)

step1_chain = step1_prompt | llm | step1_parser


# ==========================================
# STEP 2: EVALUATE CANDIDATE SKILLS
# ==========================================

class SkillEvaluation(BaseModel):
    skill: str = Field(
        description="The skill being assessed"
    )
    demonstrated: bool = Field(
        description="Whether the candidate clearly demonstrated the skill"
    )
    evidence_or_gap: str = Field(
        description="Evidence supporting the skill or explanation of the gap"
    )


class SkillEvaluations(BaseModel):
    evaluations: list[SkillEvaluation] = Field(
        description="Evaluation of each skill"
    )


step2_parser = PydanticOutputParser(
    pydantic_object=SkillEvaluations
)

step2_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a technical interviewer evaluating candidate skills.

Evaluate each requested skill using only the candidate's answers.

Instructions:
- Assess every skill provided by the user.
- Mark demonstrated as true only when clear evidence is available.
- If evidence is missing or insufficient, mark demonstrated as false.
- Explain the supporting evidence or the gap for each skill.
- Do not invent evidence or make unsupported assumptions.

Return the evaluation in the required structured format:
{format_instructions}"""
        ),
        (
            "human",
            """Candidate Answers:
{answers}

Skills to Assess:
{skills_to_assess}

Evaluate each requested skill."""
        )
    ]
)

step2_prompt = step2_prompt.partial(
    format_instructions=step2_parser.get_format_instructions()
)

step2_chain = step2_prompt | llm | step2_parser


# ==========================================
# STEP 3: GENERATE THE FINAL HIRING SCORECARD
# ==========================================

class ScorecardRow(BaseModel):
    skill: str = Field(
        description="The skill being evaluated"
    )
    demonstrated: bool = Field(
        description="Whether the skill was demonstrated"
    )
    evidence_or_gap: str = Field(
        description="Evidence or gap for the skill"
    )


class HiringScorecard(BaseModel):
    scorecard: list[ScorecardRow] = Field(
        description="List of skill evaluation rows"
    )
    overall_recommendation: str = Field(
        description="Overall recommendation based on the evaluation"
    )


step3_parser = PydanticOutputParser(
    pydantic_object=HiringScorecard
)

step3_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a hiring panel coordinator.

Create a structured hiring scorecard from the supplied skill evaluations.

Instructions:
- Include every evaluated skill.
- Include whether each skill was demonstrated.
- Include the supporting evidence or explain the gap.
- Base the overall recommendation only on the supplied evaluations.
- If evidence is insufficient, state that further assessment is needed.
- Do not invent information about the candidate.

Return the result in the required structured format:
{format_instructions}"""
        ),
        (
            "human",
            """Skill Evaluation:
{skill_evaluation}

Prepare the final hiring scorecard."""
        )
    ]
)

step3_prompt = step3_prompt.partial(
    format_instructions=step3_parser.get_format_instructions()
)

step3_chain = step3_prompt | llm | step3_parser


# ==========================================
# COLLECT USER INPUT
# ==========================================

print("\n" + "=" * 50)
print("INTERVIEW ANALYSIS AND HIRING SCORECARD")
print("=" * 50)

print("\nEnter the complete interview transcript.")
print("Type END on a separate line when finished.\n")

transcript_lines = []

while True:
    try:
        line = input()

    except EOFError:
        break

    if line.strip().upper() == "END":
        break

    transcript_lines.append(line)

transcript_text = "\n".join(transcript_lines).strip()

if not transcript_text:
    raise ValueError(
        "The interview transcript cannot be empty. Please run the program again."
    )

skills_to_assess = input(
    "\nEnter skills to assess (comma-separated): "
).strip()

if not skills_to_assess:
    raise ValueError(
        "Please enter at least one skill to assess."
    )


# ==========================================
# EXECUTE STEP 1
# ==========================================

print("\n" + "=" * 50)
print("STEP 1: EXTRACTED INTERVIEW ANSWERS")
print("=" * 50)

step1_result = step1_chain.invoke(
    {"transcript_text": transcript_text}
)

print(step1_result.model_dump_json(indent=2))


# ==========================================
# EXECUTE STEP 2
# ==========================================

print("\n" + "=" * 50)
print("STEP 2: SKILL EVALUATION")
print("=" * 50)

answers_data = [
    answer.model_dump()
    for answer in step1_result.answers
]

step2_result = step2_chain.invoke(
    {
        "answers": answers_data,
        "skills_to_assess": skills_to_assess
    }
)

print(step2_result.model_dump_json(indent=2))


# ==========================================
# EXECUTE STEP 3
# ==========================================

print("\n" + "=" * 50)
print("STEP 3: FINAL HIRING SCORECARD")
print("=" * 50)

evaluation_data = [
    evaluation.model_dump()
    for evaluation in step2_result.evaluations
]

step3_result = step3_chain.invoke(
    {
        "skill_evaluation": evaluation_data
    }
)

print(step3_result.model_dump_json(indent=2))

print("\nAnalysis completed successfully.")

