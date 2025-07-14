from typing import Annotated, Sequence, TypedDict, Literal
from operator import itemgetter
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, START, StateGraph
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Optional
from langchain.tools import Tool

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question_language: str
    agent_action: Optional[str]
    rewrite_count: int

class AgenticRAG:
    def __init__(self, vector_store):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.vector_store = vector_store.get_vector_store()
        self.graph = self._build_graph()

    def _get_retrieval_tool(self, retriever):
        """Return the AnswerGenerator tool."""
        def retriever_tool(query) -> str:
            """Retrieve relevant documents based on the query."""
            docs = retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in docs])

        return Tool(
            name="retrieverTool",
            func=retriever_tool,
            description="Use this for general knowledge questions or open-ended information about the Women's Eurocup 2025. Example: 'What can you say about Spain?' or 'Who is the coach of England?'"
        )

    def _build_graph(self):
        """
        Builds the state graph for the agentic RAG system.
        This method defines the flow of the agentic RAG process, including
        the retrieval of documents, grading of relevance, generation of responses, and default response if
        it doesnt found relevant documents.
        """
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs = {'k': 5})
        retriever_node = ToolNode([self._get_retrieval_tool(retriever)])
        graph_builder = StateGraph(State)
        graph_builder.add_node("agent", self._agent)
        graph_builder.add_node("retrieverTool", retriever_node)
        graph_builder.add_node("rewrite", self._rewrite)
        graph_builder.add_node("generate", self._generate)
        graph_builder.add_node("notfound", self._not_found)

        graph_builder.add_edge(START, "agent")

        graph_builder.add_conditional_edges( 
            "agent",
            tools_condition,
            {
                "tools": "retrieverTool",
                END: END
            }
        )
        graph_builder.add_conditional_edges(
            "retrieverTool",
            self._grade_documents
        )

        def rewrite_condition(state):
            if state.get("agent_action") == "NOT_FOUND":
                return "notfound"
            return "agent"

        graph_builder.add_conditional_edges(
            "rewrite",
            rewrite_condition
        )

        graph_builder.add_edge("generate", END)
        graph_builder.add_edge("notfound", END)

        return graph_builder.compile()
    
    def _grade_documents(self, state) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """
        class grade(BaseModel):
            """Binary score for relevance check."""
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        llm_with_structured_output = self.llm.with_structured_output(grade)
        messages = state["messages"]
        last_message = messages[-1]
        question = messages[0].content
        docs = last_message.content
  
        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        chain = prompt | llm_with_structured_output
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score

        if score == "yes" and len(docs) > 0:
            return "generate"
        else:
            return "rewrite"
        
    def _agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        messages = state["messages"]
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs = {'k': 5})
        self.tools = [self._get_retrieval_tool(retriever)]
        llm_with_tools = self.llm.bind_tools(self.tools, tool_choice="required")
        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}
    
    def _not_found(self, state):
        translate_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        rephrase_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        translation_prompt = PromptTemplate.from_template(
            "Translate this to {language} language:\n\n"
            "I don't have information about {topic}, but I can assist you with questions about the Women's Football Eurocup 2025."
        )

        rephrase_prompt = PromptTemplate.from_template(
            "Rewrite the following question as a noun phrase that describes its topic, like a short description:\n"
            "Question: {question}\n"
            "Description:"
        )
        messages = state["messages"]
        question = messages[0].content
        rephrased = rephrase_llm.invoke(rephrase_prompt.format(question=question)).content.strip()
        full_prompt = translation_prompt.format(language=state.get("question_language"), topic=rephrased)
        translated = translate_llm.invoke(full_prompt)
        return {"messages": [translated]}


    def _rewrite(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """

        count = state.get("rewrite_count", 0) + 1
        if int(count) > 2:
            state["agent_action"] = "NOT_FOUND"
            return {"agent_action" : "NOT_FOUND", "rewrite_count": count}
    
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question in the context of the woman euro cup: """,
            )
        ]

        response = self.llm.invoke(msg)
        return {"agent_action" : "agent", "messages": [response], "rewrite_count": count}

    def _generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        # Prompt
        prompt = """
                You are an assistant for question-answering tasks about the Woman Eurocup 2025. Use the following pieces of retrieved context to answer the question.
                If you don't know the answer, reply accordingly without inventing or hallucinating data and suggesting a different request or rephrase the question.
                Make sure your answer is relevant to the question and it is answered from the context only and in {language} language.
                Question: {question} 
                Context: {context} 
                Answer:
                """

        # Chain
        rag_chain = (
                    {
                        "language": itemgetter("language"), "question": itemgetter("question"), "context":  lambda x: last_message.content
                    }
                    | ChatPromptTemplate.from_template(prompt)
                    | self.llm
                    | StrOutputParser()
                    )
        # Run
        response = rag_chain.invoke({"question": question, "language": state.get("question_language")})
        return {"messages": [response]}
    