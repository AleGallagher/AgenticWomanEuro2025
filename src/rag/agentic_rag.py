import os
from operator import itemgetter
from typing import Annotated, Literal, Optional, Sequence, TypedDict

from langchain.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from rag.metadata_model import QuestionMetadataOutput
from rag.vector_stores.base_store import BaseStore

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question_language: str
    agent_action: Optional[str]
    rewrite_count: int
    question_metadata: QuestionMetadataOutput

class AgenticRAG:
    def __init__(self, vector_store: BaseStore):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.vector_store = vector_store.get_vector_store()
        self.graph = self._build_graph()
        self.retriever = None

    def _get_retrieval_tool(self):
        """Return the Retrieval tool."""
        def retriever_tool(query) -> str:
            """Retrieve relevant documents based on the query."""
            docs = self.retriever.invoke(query)
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
        it doesn't find relevant documents.
        """
        graph_builder = StateGraph(State)
        graph_builder.add_node("extractMetaData", self._extract_metadata)
        graph_builder.add_node("agent", self._agent)
        graph_builder.add_node("retrieverTool", ToolNode([self._get_retrieval_tool()]))
        graph_builder.add_node("rewrite", self._rewrite_question)
        graph_builder.add_node("generate", self._generate_response)
        graph_builder.add_node("notfound", self._not_found)

        graph_builder.add_edge(START, "extractMetaData")
        graph_builder.add_edge("extractMetaData", "agent")

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
        class Grade(BaseModel):
            """Confidence score for relevance check."""
            confidence_score: float = Field(
                description="Relevance confidence score between 0 and 1, where 1 is highly relevant."
            )

        llm_with_structured_output = self.llm.with_structured_output(Grade)
        messages = state["messages"]
        last_message = messages[-1]
        question = messages[0].content
        docs = last_message.content
  
        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing the relevance of a retrieved document to a user question. 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            Provide a confidence score between 0 and 1 to indicate how relevant the document is to the question. 
            A score closer to 1 means highly relevant, and closer to 0 means not relevant.""",
            input_variables=["context", "question"],
        )

        chain = prompt | llm_with_structured_output
        scored_result = chain.invoke({"question": question, "context": docs})
        confidence_score = scored_result.confidence_score
        relevance_threshold = float(os.getenv("RAG_RELEVANCE_THRESHOLD", 0.7))
        if len(docs) > 0 and confidence_score > relevance_threshold:
            return "generate"
        else:
            return "rewrite"

    def _extract_metadata(self, state):
        """
        Extracts metadata from the question to determine the countries involved.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with extracted metadata
        """
        question = state["messages"][0].content
        prompt = PromptTemplate(
            template="""Extract the structured data from the following question.
            Question: {question}
            """,
            input_variables=["question"],
        )
        response = self.llm.with_structured_output(QuestionMetadataOutput).invoke(prompt.format(question=question))
        return {"question_metadata": response}

    def _agent(self, state):
        """
        Invokes the agent model decide if a tool is needed based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        messages = state["messages"]
        filter_dict = {}
        if state["question_metadata"].countries:
            filter_dict = {"country": {"$in": [country.lower() for country in state["question_metadata"].countries]}}
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs = {"filter": filter_dict, 'k': int(os.getenv("RAG_RETRIEVAL_K", "5"))})
        self.tools = [self._get_retrieval_tool()]
        llm_with_tools = self.llm.bind_tools(self.tools, tool_choice="required")
        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}
    
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
        question = state["messages"][0].content
        language = state["question_language"]

        response = self.llm.invoke(combined_prompt.format(question=question, language=language))
        return {"messages": [response]}

    def _rewrite_question(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        count = state.get("rewrite_count", 0) + 1
        if int(count) > int(os.getenv("RAG_RETRY_COUNT", 2)):
            state["agent_action"] = "NOT_FOUND"
            return {"agent_action" : "NOT_FOUND", "rewrite_count": count}
    
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f"""Analyze the following question and improve it for better clarity and relevance to the Women's Eurocup 2025:
                Question:
                -------
                {question}
                -------
                Rewrite the question to make it more specific and contextually accurate.""",
            )
        ]

        response = self.llm.invoke(msg)
        return {"agent_action" : "agent", "messages": [response], "rewrite_count": count}

    def _generate_response(self, state):
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
                You are an assistant specializing in the Women's Eurocup 2025. Using the provided context, answer the question accurately and concisely in {language}.
                IMPORTANT: Only use the given context. Do not add any information not found in the context.
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

    def __call__(self, state: State):
        return self.graph.invoke(state)