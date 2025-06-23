from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

lang_detect_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
lang_prompt = PromptTemplate.from_template(
    "What is the language of the following question? Return the language name only.\n\nQuestion: {question}"
)

def detect_language(question: str) -> str:
    response = lang_detect_llm.invoke(lang_prompt.format(question=question))
    return response.content.strip()