import langchain
import os
import pandas as pd
from langchain import OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.chains.llm_symbolic_math.base import LLMSymbolicMathChain
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field

OPENAI_API_KEY = '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'
SERPAPI_API_KEY = 'insert key here'
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model_name='gpt-3.5-turbo'
)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history")

search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
llm_symbolic_math = LLMSymbolicMathChain.from_llm(llm)
file_path='/Users/evelyn/Desktop/langchain/DeepMindMathDataset.csv'
df = pd.read_csv(file_path)


class CalculatorInput(BaseModel):
    question: str = Field()

tools = [
    Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        args_schema=CalculatorInput
    )
]

tools.append(
    Tool.from_function(
        func=llm_symbolic_math.run,
        name="SymbolicMath",
        description="useful for when you need to answer math questions with symbols"
    )
)

tools.append (
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about conversion"
    ),
)

agent = initialize_agent(
    agent='conversational-react-description', 
    tools=tools, 
    llm=llm,
    verbose=True,
    max_iterations=3,
    memory=memory,
)

for i in range(1,81):
    question = df.loc[i, "Question"]
    agent.run(question)
    