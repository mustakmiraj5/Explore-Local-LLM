from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import json
import platform
import os
import psutil
import subprocess


# Ollama OpenAI-compatible client
client = OpenAI(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"]
)

MODEL = "qwen3.5:4b"

# -------------------------
# TOOL IMPLEMENTATIONS
# -------------------------

def get_current_datetime():
    now = datetime.now(ZoneInfo("Asia/Dhaka"))

    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day": now.strftime("%A"),
        "timezone": "Asia/Dhaka"
    }

def get_system_info():
    cpu = platform.processor() or "Unknown"

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=3
            )
            cpu = result.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            pass

    memory = psutil.virtual_memory()

    return {
        "os": platform.system(),
        "os_version": platform.mac_ver()[0] or platform.release(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "total_ram_gb": round(memory.total / (1024 ** 3), 2),
        "available_ram_gb": round(memory.available / (1024 ** 3), 2)
    }

def get_current_directory():
    return {
        "current_directory": os.getcwd()
    }

# WMO weather codes used by Open-Meteo
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail"
}

def get_weather(city: str = "Dhaka"):
    # Open-Meteo's geocoder matches names only, so drop ", Country" if the model adds it
    name = city.split(",")[0].strip()

    with httpx.Client(timeout=10) as http:
        places = http.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1}
        ).raise_for_status().json().get("results")

        if not places:
            raise ValueError(f"City not found: {city}")

        place = places[0]

        weather = http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
                "timezone": "auto",
                "forecast_days": 1
            }
        ).raise_for_status().json()

    current = weather["current"]
    daily = weather["daily"]

    return {
        "location": f"{place['name']}, {place.get('country', '')}",
        "local_time": current["time"],
        "current": {
            "condition": WEATHER_CODES.get(current["weather_code"], "unknown"),
            "temperature_c": current["temperature_2m"],
            "feels_like_c": current["apparent_temperature"],
            "humidity_percent": current["relative_humidity_2m"],
            "wind_kmh": current["wind_speed_10m"]
        },
        "today": {
            "condition": WEATHER_CODES.get(daily["weather_code"][0], "unknown"),
            "high_c": daily["temperature_2m_max"][0],
            "low_c": daily["temperature_2m_min"][0],
            "rain_chance_percent": daily["precipitation_probability_max"][0]
        }
    }

# -------------------------
# TOOL REGISTRY
# -------------------------

AVAILABLE_FUNCTIONS = {
    "get_current_datetime": get_current_datetime,
    "get_system_info": get_system_info,
    "get_current_directory": get_current_directory,
    "get_weather": get_weather
}

# -------------------------
# TOOL DEFINITIONS
# -------------------------

TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get current date and time in Dhaka, Bangladesh.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get computer operating system, CPU and RAM information.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_directory",
            "description": "Get the current working directory.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather and today's forecast for a city. If the user doesn't name a city, use Dhaka.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Dhaka' or 'London'."
                    }
                },
                "required": []
            }
        }
    }
]

# -------------------------
# TOOL EXECUTION
# -------------------------

def execute_tool(tool_call):

    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments or "{}")

    print(f"\n[Tool called]: {function_name}")

    function = AVAILABLE_FUNCTIONS.get(function_name)

    if function is None:
        return {"error": "Unknown function"}

    try:
        result = function(**arguments)
        print("[Tool executed successfully]")

        return result

    except Exception as error:
        return {"error": str(error)}

# -------------------------
# AGENT LOOP
# -------------------------

def chat():

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": """
            You are a helpful local AI assistant.

            You have access to tools that can retrieve information
            about the user's computer.

            Use the appropriate tools whenever necessary.
            Never invent system information or current date/time.
            """
        }
    ]

    print("Local Qwen Agent Started!")
    print("Type 'exit' to quit.\n")

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input:
            continue

        messages.append({
            "role": "user",
            "content": user_input
        })

        # Maximum 5 tool-calling rounds
        for _ in range(5):

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message

            # Save assistant response
            messages.append(assistant_message.model_dump(exclude_none=True))  # type: ignore[arg-type]

            # No more tools needed
            if not assistant_message.tool_calls:
                print(f"\nQwen: {assistant_message.content}\n")
                break

            # Execute each requested tool
            for tool_call in assistant_message.tool_calls:

                result = execute_tool(tool_call)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

        else:
            print("\nMaximum tool-calling rounds reached.\n")


if __name__ == "__main__":
    chat()
