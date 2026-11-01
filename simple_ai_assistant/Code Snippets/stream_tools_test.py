import os

from openai import OpenAI
from dotenv import load_dotenv
from types import SimpleNamespace

from tools import TOOLS

load_dotenv()

client = OpenAI(
  base_url=os.getenv("BASE_URL"),
  api_key=os.getenv("API_KEY")
)

stream = client.chat.completions.create(
  model=os.getenv("LLM_MODEL"),
  messages=[{"role": "user", "content": "Tell me the weather of Patna and weather of Delhi."}],
  tools=TOOLS,
  stream=True,
)

full_response = ""
calls = {} # index -> tool call being built

for chunk in stream:
  if not chunk.choices:
    continue
  
  delta = chunk.choices[0].delta

  for tool_call in (delta.tool_calls or []):
    obj = calls.setDefault(
      tool_call.index,
      SimpleNamespace(id=None, function=SimpleNamespace(name="", arguments="")),
    )

    if tool_call.id:
      obj.id = tool_call.id
    if tool_call.function and tool_call.function.name:
      obj.function.name += tool_call.function.name
    if tool_call.function and tool_call.function.arguments:
      obj.function.arguments += tool_call.function.arguments

calls_list = []
for key in sorted(calls):
  calls_list.append(calls[key])
      