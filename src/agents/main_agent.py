from langgraph.graph import StateGraph
import sys
sys.path.append('f:/Python/AgenticEuro2025/src/')
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from tools.sql_tool import get_sql_tool
from tools.agentic_rag_tool import agentic_rag_stream
from tools.qualification_tool import get_qualification_options
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from services.database_service import DatabaseService
import os
from langgraph.graph import END
os.environ['OPENAI_API_KEY'] = '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'
from langchain.prompts import PromptTemplate

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question_language: str
    selected_tool: str
    user_id: str
    country: str

class MainAgent:
    def __init__(self, model, vector_store):
        self.tools = self._get_tools()
        self.llm = model.bind_tools(self.tools, tool_choice="required")
        self.model = model
        self.graph = self._build_graph()
        self.vector_store = vector_store
        self.database_service = DatabaseService()

    def _get_tools(self):
        """Bind tools to the LLM."""
        tools = [
            Tool.from_function(
                name="SQLQueryTool",
                description="Use for specific data such as coach names, team information, player lists, scores, and matches. Example: 'Who is the coach of England?'",
                func=lambda **kwargs: "placeholder",  # this won't be called; just for LLM awareness
            ),
            Tool.from_function(
                name="agentic_rag",
                description="Use this for general knowledge questions or open-ended information about the Women's Eurocup 2025. Example: 'What can you say about Spain?' or 'Is there VAR?' or 'When will start the cup?'",
                func=lambda **kwargs: "placeholder"
            ),
            Tool.from_function(
                name="qualification_tool",
                description="Use to know what need a team to qualify, pass to the next stage. Example: 'What need Spain to qualify?', or 'What need Spain to qualify to the next stage?'",
                func=lambda **kwargs: "placeholder"
            )
        ]
        return tools

    def _agent_node(self, state):
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _tool_executor(self, state):
        tool_call = state["messages"][-1].tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "SQLQueryTool":
            tool_func = get_sql_tool.invoke({"model": self.model,
                 "agent_input": state["messages"][0].content,
                "question_language": state["question_language"]}
            )
        elif tool_name == "agentic_rag":
            tool_func = agentic_rag_stream.invoke({
                "vector_store": self.vector_store,
                "question": state["messages"][0].content,
                "question_language": state["question_language"]
            })
        elif tool_name == "qualification_tool":
            tool_func = get_qualification_options.invoke({
                "model": self.model,
                "agent_input": state["messages"][0].content,
                "question_language": state["question_language"]
            })
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        result = tool_func
        self.database_service.save_question_answer(
            user_id =  state["user_id"],
            country = state["country"],
            question =  state["messages"][0].content,
            response = result,
            question_language = state["question_language"],
            tool = tool_name,
        )
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call["id"])], "selected_tool": tool_name}

    def _build_graph(self):
        def detect_language_node(state):
            question = state["messages"][0].content
            detected_language = self._detect_language(question)
            return {"question_language": detected_language}

        graph = StateGraph(State)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tool_executor", self._tool_executor)
        graph.add_node("detect_language", detect_language_node)

        # Routing logic
        def route(state):
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tool_executor"
            return END

        graph.set_entry_point("detect_language")  # Set the entry point to the new node
        graph.add_edge("detect_language", "agent") 
        graph.add_conditional_edges("agent", route, {
            "tool_executor": "tool_executor",
            END: END
        })
        graph.add_edge("tool_executor", END)
        runnable = graph.compile()
        return runnable

    def _detect_language(self, question: str) -> str:
        lang_detect_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        lang_prompt = PromptTemplate.from_template(
            "What is the language of the following question? Return the language name only.\n\nQuestion: {question}"
        )
        response = lang_detect_llm.invoke(lang_prompt.format(question=question))
        return response.content.strip()