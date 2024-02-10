import gradio as gr
from langchain.chains import create_tagging_chain_pydantic
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
import os
import dotenv
dotenv.load_dotenv()

class Tags(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )
def tagging(file_paths):
  llm = ChatOpenAI(temperature=0, model="gpt-4")
  chain = create_tagging_chain_pydantic(Tags, llm)
  docs = ""
  for path in file_paths:
    head = os.path.split(path)
    loader = PyPDFLoader(path)
    doc = loader.load()
    docs += str(head[1])+": "+str(chain.run(doc))+"\n\n"
  return docs

entity = gr.Interface(
    tagging,
    [
    gr.File(label="Files", file_count="multiple"),
    ],
    "textbox",
    title="Tag the given Documents using Langchain and OpenAI's GPT-4",
    theme = "gradio/monochrome"
)
entity.launch()