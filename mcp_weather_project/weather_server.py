
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Weather Server")


# Expose a Python function as an MCP tool
@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""

    # Sample data for learning
    return f"The weather in {city} is 30°C and sunny."


if __name__ == "__main__":
    mcp.run(transport="stdio")