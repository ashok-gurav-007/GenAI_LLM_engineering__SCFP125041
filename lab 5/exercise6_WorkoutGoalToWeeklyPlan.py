
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)


# ==========================================
# STEP 1: EXTRACT FITNESS GOAL DETAILS
# ==========================================

class FitnessGoal(BaseModel):
    goal_type: str = Field(
        description="Fitness goal of the person"
    )
    fitness_level: str = Field(
        description="Current fitness level"
    )
    days_available: int = Field(
        description="Number of workout days available per week"
    )
    equipment_access: str = Field(
        description="Available workout equipment"
    )


step1_parser = PydanticOutputParser(
    pydantic_object=FitnessGoal
)


step1_prompt = PromptTemplate(
    template="""
Analyze the following fitness goal description.

Extract:
- Goal type
- Current fitness level
- Days available per week
- Equipment access

Use the separately provided number of available workout days
as the authoritative value for days_available.

If fitness level or equipment access is not specified,
state that it is not specified. Do not invent personal details.

Fitness Goal Description:
{fitness_goal_text}

Days Available per Week:
{days_per_week}

{format_instructions}
""",
    input_variables=[
        "fitness_goal_text",
        "days_per_week"
    ],
    partial_variables={
        "format_instructions": step1_parser.get_format_instructions()
    }
)


step1_chain = step1_prompt | llm | step1_parser


# ==========================================
# STEP 2: GENERATE WEEKLY WORKOUT PLAN
# ==========================================

class WorkoutDay(BaseModel):
    day: str = Field(
        description="Day of the week"
    )
    workout: str = Field(
        description="Workout planned for the day"
    )
    duration: str = Field(
        description="Workout duration"
    )
    notes: str = Field(
        description="Additional workout notes"
    )


class WeeklyWorkoutPlan(BaseModel):
    weekly_plan: list[WorkoutDay] = Field(
        description="Day-by-day weekly workout plan"
    )


step2_parser = PydanticOutputParser(
    pydantic_object=WeeklyWorkoutPlan
)


step2_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a fitness planning assistant.

Create a structured weekly workout plan using only the
extracted fitness details provided.

Requirements:
- Match the plan to the person's fitness goal and fitness level.
- Create workout sessions for exactly the number of days available.
- Use the stated equipment access.
- Include appropriate rest days in the weekly schedule.
- Include a workout, duration, and notes for each planned session.
- Keep the plan practical and safe for the stated fitness level.
- Include warm-up and cool-down guidance in the notes.
- Do not invent medical information.
- If the fitness level or equipment access is unspecified,
  choose a conservative beginner-friendly approach.

{format_instructions}
"""
    ),
    (
        "user",
        """
Goal Type: {goal_type}
Fitness Level: {fitness_level}
Days Available: {days_available}
Equipment Access: {equipment_access}

Create the weekly workout plan.
"""
    )
])


step2_prompt = step2_prompt.partial(
    format_instructions=step2_parser.get_format_instructions()
)


step2_chain = step2_prompt | llm | step2_parser


# ==========================================
# USER INPUT
# ==========================================

print("\n" + "=" * 50)
print("WORKOUT GOAL TO WEEKLY PLAN PIPELINE")
print("=" * 50)

fitness_goal_text = input(
    "\nEnter your fitness goal description: "
).strip()

if not fitness_goal_text:
    raise ValueError("Fitness goal description cannot be empty.")


while True:
    days_per_week = input(
        "Enter number of days available per week (1-7): "
    ).strip()

    try:
        days_per_week_int = int(days_per_week)

        if 1 <= days_per_week_int <= 7:
            break

        print("Please enter a number between 1 and 7.")

    except ValueError:
        print("Invalid input. Please enter a whole number.")


# ==========================================
# RUN STEP 1
# ==========================================

print("\nSTEP 1: Extracting Fitness Goal Details...")

step1_result = step1_chain.invoke({
    "fitness_goal_text": fitness_goal_text,
    "days_per_week": str(days_per_week_int)
})

print("\nSTEP 1: Extracted Goal Details")
print(step1_result.model_dump_json(indent=2))


# ==========================================
# RUN STEP 2
# ==========================================

print("\nSTEP 2: Generating Weekly Workout Plan...")

step2_result = step2_chain.invoke({
    "goal_type": step1_result.goal_type,
    "fitness_level": step1_result.fitness_level,
    "days_available": step1_result.days_available,
    "equipment_access": step1_result.equipment_access
})

print("\nSTEP 2: Weekly Workout Plan")
print(step2_result.model_dump_json(indent=2))


# ==========================================
# COMPLETION
# ==========================================

print("\n" + "=" * 50)
print("Workout planning completed successfully.")
print("=" * 50)
