import langchain
import os
import pandas as pd
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

OPENAI_API_KEY = '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'


from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history")
file_path='/Users/evelyn/Desktop/langchain/DeepMindMathDataset.csv'
df = pd.read_csv(file_path)
agent_executor = create_python_agent(
    llm=OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
    tool=PythonREPLTool(),
    verbose=True,
    agent='conversational-react-description',
    memory=memory
)

for i in range(82,91):
    question = df.loc[i, "Question"]
    agent_executor.run(question)