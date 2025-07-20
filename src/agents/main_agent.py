from langgraph.graph import StateGraph
from langchain_core.messages import ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from tools.sql_tool import get_sql_tool
from tools.agentic_rag_tool import agentic_rag_stream
from tools.qualification_tool import get_qualification_options
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from services.database_service import DatabaseService
from langgraph.graph import END
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableLambda

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
        """Return the tools to bind to the LLM."""
        tools = [get_sql_tool,
                agentic_rag_stream,
                get_qualification_options
                ]
        return tools

    def _get_dict_tools(self):
        """Return a dictionary of tools."""
        tools = {
            "SQLQueryTool": get_sql_tool,
            "agentic_rag": agentic_rag_stream,
            "qualification_tool": get_qualification_options
        }
        return tools

    def _agent_node(self, state):
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _translate_question(self, question: str) -> str:
        """Translate the question to the specified language."""
        translation_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        translation_prompt = PromptTemplate.from_template(
            """Translate the following question to English.
        ⚠️  DO NOT translate names of people, clubs, stadiums, or cities.
        ✅  DO translate country or national team names into their English form
            (e.g., 'España' → 'Spain', 'Alemania' → 'Germany').
            Return only the translated question.:\n\n{question}"""
        )
        translated_question = translation_llm.invoke(translation_prompt.format(question=question))
        return translated_question.content.strip()
    
    def _tool_executor(self, state):
        tool_call = state["messages"][-1].tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"].copy()
        original_question = tool_args.get("question")

        tool_args["question"] = self._translate_question(original_question) if state["question_language"] != "English" else original_question
        tool_args["model"] = self.model
        tool_args["question_language"] = state["question_language"]
        tool_args["vector_store"] = self.vector_store
        print("translated question", tool_args["question"])

        result = self._get_dict_tools()[tool_name].invoke(tool_args)
        self.database_service.save_question_answer(
            user_id =  state["user_id"],
            country = state["country"],
            question =  tool_args.get("question"),
            original_question = original_question,
            response = result,
            question_language = state["question_language"],
            tool = tool_name,
        )
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call["id"])], "selected_tool": tool_name}

    def _build_graph(self):
        def detect_language_node(state):
            question = state["messages"][-1].content
            detected_language = self._detect_language(question)
            return {"question_language": detected_language}

        graph = StateGraph(State)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tool_executor", self._tool_executor)
        graph.add_node("detect_language", detect_language_node)
        graph.add_node("summarize", self._summarize_stats_node)

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
        def needs_summary(state):
            last_user_input = state["messages"][-3].content.lower()
            if "write" in last_user_input or "text" in last_user_input or "summary" in last_user_input:
                return "summarize"
            return END
        graph.add_conditional_edges("tool_executor", needs_summary)
        graph.add_edge("summarize", END)
        graph.add_edge("tool_executor", END)
        memory = MemorySaver()
        runnable = graph.compile(checkpointer=memory)
        return runnable

    def _summarize_stats_node(self, state):
        def summarize(state):
            messages = state["messages"]
            sql_output = messages[-1].content
            prompt = f"Write a summary of the following football match statistics in a natural and concise way:\n\n{sql_output}"
            summary = self.model.invoke(prompt)
            return {"messages": summary}

        return RunnableLambda(summarize)

    def _detect_language(self, question: str) -> str:
        lang_detect_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        lang_prompt = PromptTemplate.from_template(
            "What is the language of the following question? Return the language name only.\n\nQuestion: {question}"
        )
        response = lang_detect_llm.invoke(lang_prompt.format(question=question))
        return response.content.strip()