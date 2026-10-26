
from mcp.server.fastmcp import FastMCP

# Create the MCP tool provider
mcp = FastMCP("WeatherTools")


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the weather information for a city."""

    weather_data = {
        "hyderabad": "30°C and sunny",
        "delhi": "32°C and clear",
        "mumbai": "29°C and cloudy",
    }

    return weather_data.get(
        city.lower(),
        f"Weather information is not available for {city}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
