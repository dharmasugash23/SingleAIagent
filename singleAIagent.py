import os
import certifi
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# --------------------------------------------------
# 2. Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Agentic AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI Assistant")
st.markdown("Search + Weather AI Agent using LangChain and Groq")


# --------------------------------------------------
# 3. Check API keys
# --------------------------------------------------

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing in .env")

if not TAVILY_API_KEY:
    st.warning("TAVILY_API_KEY is missing in .env")

if not WEATHERSTACK_API_KEY:
    st.warning("WEATHERSTACK_API_KEY is missing in .env")


# --------------------------------------------------
# 4. Weather Tool
# --------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather of a city using Weatherstack API.
    """

    if not WEATHERSTACK_API_KEY:
        return "Weatherstack API key is not configured."

    url = (
        f"http://api.weatherstack.com/current"
        f"?access_key={WEATHERSTACK_API_KEY}"
        f"&query={city}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        if "error" in data:
            return f"Weather API error: {data['error']['info']}"

        current = data.get("current", {})

        temperature = current.get("temperature")
        description = current.get("weather_descriptions", [])

        if temperature is None:
            return "Weather information is not available."

        description = ", ".join(description)

        return (
            f"Current weather in {city}: "
            f"{temperature}°C, {description}"
        )

    except requests.RequestException as e:
        return f"Unable to fetch weather: {e}"


# --------------------------------------------------
# 5. Tavily Search Tool
# --------------------------------------------------

search_tool = TavilySearch(
    max_results=5
)


# --------------------------------------------------
# 6. Create Tools List
# --------------------------------------------------

tools = [
    search_tool,
    get_weather
]


# --------------------------------------------------
# 7. Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-120b"
)


# --------------------------------------------------
# 8. Create Agent
# --------------------------------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a helpful AI assistant.

Rules:
1. Use the weather tool when the user asks about weather.
2. Use the Tavily search tool when current web information is needed.
3. For normal questions, answer directly.
4. Give clear and concise answers.
5. Do not mention internal tool-calling details to the user.
"""
)


# --------------------------------------------------
# 9. User Input
# --------------------------------------------------

user_input = st.text_input(
    "Enter your query:",
    placeholder="Tell me the weather in New York and what is AI"
)


# --------------------------------------------------
# 10. Run Agent
# --------------------------------------------------

if st.button("🚀 Run Agent"):

    if not user_input:
        st.warning("Please enter a query.")

    else:

        with st.spinner("Agent is working..."):

            try:

                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_input
                            }
                        ]
                    }
                )

                # Get final AI message
                final_message = response["messages"][-1]

                st.success("Agent completed successfully!")

                st.markdown("### 🤖 Agent Response")

                st.write(final_message.content)

            except Exception as e:

                st.error("An error occurred:")
                st.exception(e)