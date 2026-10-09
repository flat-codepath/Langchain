import requests

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


# --------------------------------------------------
# 1. Create the LLM
# --------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# --------------------------------------------------
# 2. Create the Weather Tool
# --------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.
    """

    # ----------------------------------------------
    # Step 1: Convert city name to latitude/longitude
    # ----------------------------------------------

    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"

    geocoding_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(
        geocoding_url,
        params=geocoding_params
    )

    response.raise_for_status()

    location_data = response.json()

    if not location_data.get("results"):
        return f"Could not find the city: {city}"

    location = location_data["results"][0]

    latitude = location["latitude"]
    longitude = location["longitude"]

    actual_city = location["name"]
    country = location.get("country", "")

    # ----------------------------------------------
    # Step 2: Call Weather API
    # ----------------------------------------------

    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "auto"
    }

    response = requests.get(
        weather_url,
        params=weather_params
    )

    response.raise_for_status()

    weather_data = response.json()

    current = weather_data["current"]

    # ----------------------------------------------
    # Step 3: Return useful data to the LLM
    # ----------------------------------------------

    return (
        f"City: {actual_city}, {country}\n"
        f"Temperature: {current['temperature_2m']} °C\n"
        f"Humidity: {current['relative_humidity_2m']}%\n"
        f"Wind speed: {current['wind_speed_10m']} km/h"
    )


# --------------------------------------------------
# 3. Give the tool to the LLM
# --------------------------------------------------

llm_with_tools = llm.bind_tools([get_weather])


# --------------------------------------------------
# 4. Terminal Chat
# --------------------------------------------------

print("Weather AI")
print("Type 'exit' to stop.")
print()

while True:

    user_question = input("You: ")

    if user_question.lower() == "exit":
        break

    # ----------------------------------------------
    # First LLM call
    # ----------------------------------------------

    response = llm_with_tools.invoke(user_question)

    # ----------------------------------------------
    # Check whether LLM wants to call a tool
    # ----------------------------------------------

    if response.tool_calls:

        for tool_call in response.tool_calls:

            if tool_call["name"] == "get_weather":

                city = tool_call["args"]["city"]

                # Execute the actual Python function
                weather_result = get_weather.invoke(
                    tool_call["args"]
                )

                # ----------------------------------
                # Send weather result back to LLM
                # ----------------------------------

                final_response = llm.invoke(
                    [
                        (
                            "system",
                            "Answer the user's question using the weather data provided."
                        ),
                        (
                            "human",
                            user_question
                        ),
                        (
                            "tool",
                            weather_result
                        )
                    ]
                )

                print("AI:", final_response.content)

    else:

        print("AI:", response.content)