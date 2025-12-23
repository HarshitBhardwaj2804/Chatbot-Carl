# search_engine.py
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from dotenv import load_dotenv

groq_api_key = os.getenv("GROQ_API_KEY_MAIN_PROJECT")
tavily_api_key = os.getenv("TAVILY_API_KEY")

class TavilyAgentwithHistory:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.llm = ChatGroq(model=model, groq_api_key=groq_api_key, temperature=0.3)

        # Tools
        self.tools = [
            TavilySearchResults(max_results=2, tavily_api_key=tavily_api_key)
        ]

        # Prompt Template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an intelligent assistant whose task is to answer the user query "
                "by searching it over the web using the provided tools. "
                "DO NOT answer from your own knowledge. "
                "ALWAYS invoke the search tool to get the answer from the web."
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create the agent
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            prompt=self.prompt,
            tools=self.tools
        )

        # Create executor
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

class TavilyAgent:
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.llm = ChatGroq(model=model, groq_api_key=groq_api_key, temperature=0.3)

        # Tools
        self.tools = [
            TavilySearchResults(max_results=2, tavily_api_key=tavily_api_key)
        ]

        # Prompt Template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an intelligent assistant whose task is to answer the user query "
                "by searching it over the web using the provided tools. "
                "DO NOT answer from your own knowledge. "
                "ALWAYS invoke the search tool to get the answer from the web."
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create the agent
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            prompt=self.prompt,
            tools=self.tools
        )

        # Create executor
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

