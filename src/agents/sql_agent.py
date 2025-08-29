import os
import time
from typing import Annotated, Sequence, TypedDict

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from services.prompt_utils import PromptUtils

_cached_toolkit = None
_cached_prompt = None
_cache_last_updated = None
CACHE_REFRESH_INTERVAL = 900

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question_language: str
    input: str

class SQLAgent:
    def __init__(self, llm):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.graph = self._get_graph_executor()
    
    def _get_graph_executor(self):
        builder = StateGraph(State)
        db_agent = self._create_reasoning_node()
        builder.add_node("agent", db_agent)
        builder.add_node("notfound", self._not_found)

        def relevant_answer(state):
            if "no results found" in state["messages"][-1].content.lower():
                return "notfound"
            return END

        builder.add_conditional_edges(
            "agent",
            relevant_answer
        )
        builder.set_entry_point("agent")
        builder.add_edge("notfound", END)
        return builder.compile()
    
    def _not_found(self, state):
        """
        Handles the case where no relevant information is found.

        This method generates a response when the system cannot find relevant documents
        or information for the user's query. It rewrites the user's question into a noun phrase
        that describes its topic, translates the response into the user's preferred language,
        and provides a fallback message.

        Args:
            state (dict): The current state of the system, which includes:
                - messages: A list of messages in the conversation.
                - question_language: The language in which the response should be generated.

        Returns:
            dict: The updated state with a fallback response appended to the messages.
        """

        combined_prompt = PromptTemplate.from_template(
        """
        Rewrite the following question as a noun phrase that describes its topic, like a short description.
        Then translate the following response into {language}:
        
        Question: {question}
        Description: <rephrased topic>
        
        Response: "I don't have information about <rephrased topic>, but I can assist you with questions about the Women's Football Eurocup 2025."
        Translated Response:
        """
        )
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        response = llm.invoke(combined_prompt.format(question=state["input"], language=state["question_language"]))
        return {"messages": [response]}

    def _create_reasoning_node(self):
        """
        Creates the SQL reasoning node that converts natural language questions 
        into database queries and executes them.
        
        Sets up an OpenAI functions agent with SQL database tools to handle
        football-related queries. The agent follows a 5-step process: plan,
        generate SQL, self-check, execute, and format results.
        
        Returns:
            callable: Function that processes agent state and returns SQL query results.
                    Handles errors gracefully and provides user-friendly messages.
        
        Note:
            Uses cached toolkit for performance. Max 10 iterations to prevent loops.
        """
        tools, prompt = self._setup_sql_toolkit()
        agent = create_openai_functions_agent(llm=self.llm, tools=tools, prompt=prompt)
        executor = AgentExecutor(agent=agent, tools=tools, max_iterations=10, handle_parsing_errors=True)

        def run_agent(state: State) -> dict:
            try:
                result = executor.invoke({"input": state["input"], "language": state["question_language"]})
                if "agent stopped due to iteration limit or time limit." in result["output"].lower():
                    return {"messages": [AIMessage(content="Seems that there are no results for this question. Can I help you with something else?")]}
                return {"messages": [AIMessage(content=result["output"])]}
            except Exception as e:
                print(f"Error in SQL Agent: {str(e)}")
                error_message = f"An error occurred while processing your request. Please try a different request or rephrase your question."
                return {"messages": [AIMessage(content=error_message)]} 
        return run_agent
    
    def _setup_sql_toolkit(self):
        """
        Sets up and caches the SQL database toolkit and prompt for football queries.
        
        Creates a PostgreSQL connection with custom table schemas and a specialized
        prompt that enforces the 5-step query process (PLAN → SQL → SELF-CHECK → 
        EXECUTE → RESULT). Includes caching to improve performance.
        
        Returns:
            tuple: (tools, prompt) where tools is a list of SQL database tools
                and prompt is the ChatPromptTemplate for the agent.
        
        Note:
            Cache refreshes every 15 minutes. Handles connection errors gracefully.
        """
        global _cached_toolkit, _cached_prompt, _cache_last_updated

        def initialize_cache():
            prompt_config = PromptUtils.load_prompt_template("sql_agent")
            """Helper function to initialize the cache."""
            db = SQLDatabase.from_uri(
                    os.getenv("POSTGRES_HOST"),
                    view_support=True,
                    include_tables=prompt_config["include_tables"],
                    sample_rows_in_table_info=2,
                    custom_table_info=prompt_config["custom_table_info"]
            )
            toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
            prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(prompt_config["system_message"]),
                    HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}")
            ])

            return toolkit.get_tools(), prompt
         
        try:
            # Check if cache needs to be refreshed
            current_time = time.time()
            if (_cached_toolkit and _cached_prompt and 
                _cache_last_updated and 
                current_time - _cache_last_updated < CACHE_REFRESH_INTERVAL):
                return _cached_toolkit, _cached_prompt
            
            # Update cache
            _cached_toolkit, _cached_prompt = initialize_cache()
            _cache_last_updated = current_time
            return _cached_toolkit, _cached_prompt
        except Exception as e:
            print(f"Error initializing SQL toolkit or prompt: {e}")
            _cached_toolkit, _cached_prompt = initialize_cache()
            _cache_last_updated = current_time
            return _cached_toolkit, _cached_prompt
        
    def __call__(self, state: State):
        return self.graph.invoke(state)