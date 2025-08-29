from typing import Annotated, Sequence, TypedDict

from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from services.database_service import DatabaseService
from tools.agentic_rag_tool import agentic_rag
from tools.qualification_tool import get_qualification_options
from tools.sql_tool import get_sql_tool
from services.prompt_utils import PromptUtils

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question_language: str
    selected_tool: str
    user_id: str
    country: str
    is_valid_question: bool

class MainAgent:
    def __init__(self, model, vector_store):
        self.tools = self._get_tools()
        self.llm = model.bind_tools(self.tools)
        self.model = model
        self.graph = self._build_graph()
        self.vector_store = vector_store
        self.database_service = DatabaseService()

    def _get_tools(self):
        """Return the tools to bind to the LLM."""
        tools = [get_sql_tool,
                agentic_rag,
                get_qualification_options
                ]
        return tools

    def _get_dict_tools(self):
        """Return a dictionary of tools."""
        tools = {
            "SQLQueryTool": get_sql_tool,
            "agentic_rag": agentic_rag,
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
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for tool_call in tool_calls:
            try:
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
                results.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            except Exception as e:
                print(f"Error executing tool {tool_call['name']}: {e}")
                results.append(ToolMessage(content=f"Error : {e}", tool_call_id=tool_call["id"], name=tool_call["name"]))
        return {"messages": results}

    def _build_graph(self):
        def validate_question_node(state):
            question = state["messages"][-1].content
            is_valid = self._validate_football_question(question)
            
            if not is_valid:
                rejection_message = AIMessage(
                    content="I'm a football statistics assistant specialized in UEFA Euro championships. I can only help with football-related questions about players, teams, matches, statistics, and tournaments. Please ask me something about football!"
                )
                return {"messages": [rejection_message], "is_valid_question": False}
            
            return {"is_valid_question": True}

        def detect_language_node(state):
            question = state["messages"][-1].content
            detected_language = self._detect_language(question)
            return {"question_language": detected_language}

        graph = StateGraph(State)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tool_executor", self._tool_executor)
        graph.add_node("detect_language", detect_language_node)
        graph.add_node("validate_question", validate_question_node)

        # Routing logic
        def route(state):
            last = state["messages"][-1]
            if len(last.tool_calls) > 0:
                return "tool_executor"
            return END
        def route_from_validation(state):
            if state.get("is_valid_question", True):
                return "agent"
            return END
        graph.set_entry_point("detect_language")  # Set the entry point to the new node
        graph.add_edge("detect_language", "validate_question") 
        graph.add_conditional_edges("agent", route, {
            "tool_executor": "tool_executor",
            END: END
        })
        graph.add_conditional_edges("validate_question", route_from_validation, {
        "agent": "agent",
        END: END
        })
        graph.add_edge("tool_executor", "agent")
        memory = MemorySaver()
        runnable = graph.compile(checkpointer=memory)
        return runnable

    def _detect_language(self, question: str) -> str:
        lang_detect_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        lang_prompt = PromptTemplate.from_template(
            "What is the language of the following question? Return the language name only.\n\nQuestion: {question}"
        )
        response = lang_detect_llm.invoke(lang_prompt.format(question=question))
        return response.content.strip()
    
    def _validate_football_question(self, question: str) -> bool:
        """Validate if the question is related to football/soccer."""
        validation_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt_config = PromptUtils.load_prompt_template("validation_question", "v0") 
        validation_prompt = PromptTemplate.from_template(prompt_config["template"])
        response = validation_llm.invoke(validation_prompt.format(question=question))
        return response.content.strip().upper() == "YES"
    
    def __call__(self, state: State, config):
        return self.graph.invoke(state, config)