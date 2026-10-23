
import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field

# INITIALIZATION
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=groq_api_key
)


# STEP 1: EXTRACT EVENT DETAILS
class EventDetails(BaseModel):
    event_type: str = Field(
        description="Type of event being planned"
    )
    event_date: str = Field(
        description="Date of the event in YYYY-MM-DD format"
    )
    location: str = Field(
        description="Location where the event will take place"
    )
    indoor_outdoor_preference: str = Field(
        description="Whether the event is preferred indoors or outdoors"
    )


step1_parser = PydanticOutputParser(
    pydantic_object=EventDetails
)

step1_prompt = PromptTemplate(
    template="""Analyze the event description and extract the following:
1. Event type
2. Event date in YYYY-MM-DD format
3. Event location
4. Indoor or outdoor preference

Use only the information provided.
If any detail is missing, indicate that it is not specified.
Do not invent a date or location.

Event Description:
{event_description_text}

{format_instructions}
""",
    input_variables=["event_description_text"],
    partial_variables={
        "format_instructions": step1_parser.get_format_instructions()
    }
)

step1_chain = step1_prompt | llm | step1_parser

# STEP 2: FETCH WEATHER INFORMATION
class WeatherData(BaseModel):
    temperature: str = Field(
        description="Average forecast temperature in Celsius"
    )
    precipitation_chance: str = Field(
        description="Chance of rain as a percentage"
    )
    conditions: str = Field(
        description="Expected weather conditions"
    )


def get_weather(location, event_date):
    weather_api_key = os.getenv("WEATHER_API_KEY")

    if not weather_api_key:
        raise ValueError(
            "WEATHER_API_KEY not found. Check your .env file."
        )

    url = "https://api.weatherapi.com/v1/forecast.json"

    params = {
        "key": weather_api_key,
        "q": location,
        "dt": event_date
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if "forecast" not in data:
        raise ValueError(
            "Weather forecast is unavailable for the requested date."
        )

    forecast_day = data["forecast"]["forecastday"][0]["day"]

    weather = WeatherData(
        temperature=f"{forecast_day['avgtemp_c']} °C",
        precipitation_chance=(
            f"{forecast_day['daily_chance_of_rain']}%"
        ),
        conditions=forecast_day["condition"]["text"]
    )

    return weather

# STEP 3: GENERATE WEATHER-BASED RECOMMENDATION
step3_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an event planning assistant.

Recommend whether the event should:
- Proceed as planned
- Be rescheduled
- Be moved indoors

Consider the event type, indoor/outdoor preference,
temperature, probability of rain, and weather conditions.

Explain the recommendation briefly.
Do not claim certainty about the weather.
If the weather is unsuitable for an outdoor event,
suggest a safer alternative."""
        ),
        (
            "human",
            """Event Type: {event_type}
Event Date: {event_date}
Location: {location}
Preference: {indoor_outdoor_preference}

Average Temperature: {temperature}
Chance of Rain: {precipitation_chance}
Weather Conditions: {conditions}

Provide the final weather-adjusted event recommendation."""
        )
    ]
)

step3_parser = StrOutputParser()

step3_chain = step3_prompt | llm | step3_parser

# RUN THE COMPLETE PIPELINE
print("\n" + "=" * 50)
print("WEATHER-ADJUSTED EVENT PLANNER")
print("=" * 50)

event_description_text = input(
    "\nEnter the event description: "
).strip()

if not event_description_text:
    raise ValueError("Event description cannot be empty.")


# Execute Step 1
print("\nSTEP 1: Extracting Event Details...")

step1_result = step1_chain.invoke(
    {"event_description_text": event_description_text}
)

print("\nSTEP 1: Event Details")
print(step1_result.model_dump_json(indent=2))


# Validate extracted information
if (
    not step1_result.location
    or not step1_result.event_date
    or "not specified" in step1_result.location.lower()
    or "not specified" in step1_result.event_date.lower()
):
    raise ValueError(
        "Please provide a valid event date and location in your description."
    )


# Execute Step 2
print("\nSTEP 2: Fetching Weather Information...")

try:
    step2_result = get_weather(
        step1_result.location,
        step1_result.event_date
    )

except requests.RequestException as error:
    raise RuntimeError(
        f"Unable to retrieve weather information: {error}"
    ) from error

except (KeyError, IndexError, ValueError) as error:
    raise RuntimeError(
        f"Unable to process the weather forecast: {error}"
    ) from error

print("\nSTEP 2: Weather Data")
print(step2_result.model_dump_json(indent=2))


# Execute Step 3
print("\nSTEP 3: Generating Event Recommendation...")

step3_result = step3_chain.invoke(
    {
        "event_type": step1_result.event_type,
        "event_date": step1_result.event_date,
        "location": step1_result.location,
        "indoor_outdoor_preference": (
            step1_result.indoor_outdoor_preference
        ),
        "temperature": step2_result.temperature,
        "precipitation_chance": (
            step2_result.precipitation_chance
        ),
        "conditions": step2_result.conditions
    }
)

print("\nSTEP 3: Final Recommendation")
print(step3_result)

print("\nEvent planning analysis completed successfully.")

