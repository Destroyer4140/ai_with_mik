# standard library imports
import os

# third party imports
from openai import OpenAI
from dotenv import load_dotenv
from types import SimpleNamespace

# local modules
from Utilities.utilities import log
from tools import TOOLS, run_tool

load_dotenv();

client = OpenAI(
  base_url=os.getenv("BASE_URL"),
  api_key=os.getenv("API_KEY")
)

messages = []
MAX_STEP = 5

messages.append(
  {
    "role": "system",
    "content": (
      "You are a helpful, humble assistant."
      "Answer general questions (programming , science, advice) from your own knowledge."
      "Use a tool ONLY when the user asks fro the current time, the weather, or their city."
      "Keeps answer under 100 words."
    )
  }
)

def stream_chat(history):
  stream = client.chat.completions.create(
    model=os.getenv("LLM_MODEL"),
    messages=history,
    tools=TOOLS,
    stream=True,
  )

  full_response = ""
  calls = {}
  label_printed = False

  # Streaming Stared
  try:
    for chunk in stream:
      if not chunk.choices:
        continue

      delta = chunk.choices[0].delta

      #1. Content Streaming: Do we have message (content) from the LLM
      if delta.content:
        if not label_printed:
          print("🤖 Assistant: ", end="", flush=True)
          label_printed = True
        print(delta.content, end="", flush=True)
        full_response += delta.content
      
      #2. Tool Calls
      for tool_call in  (delta.tool_calls or []):
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
  except KeyboardInterrupt as e:
    print("\n  {Stopped By You.}")

  if label_printed:
    print() # move to the next line after the assistant's response
  calls_list = []
  for key in sorted(calls):
    calls_list.append(calls[key])
  return full_response, calls_list

def build_tool_calls_payload(tool_calls):
  """
  Converts out collected tool calls into the dictonary format that the API expects inside an assistant message
  """

  payload = []
  for tool_call in tool_calls:
    tool_call_dict = {
      "id": tool_call.id,
      "type": "function",
      "function": {
        "name": tool_call.function.name,
        "arguments":  tool_call.function.arguments,
      }
    }
    payload.append(tool_call_dict)

  return payload


while True: 
  user_input = input("🧔🏻You: ")
  if user_input.lower() in ["exit", "quit", "bye", "goodbye"]: 
    break

  messages.append(
    {
      "role": "user",
      "content": user_input
    }
  )

  for step in range(MAX_STEP):
    
    full_response, tool_calls = stream_chat(messages)

    # msg = response.choices[0].message

    if not tool_calls: 
      break
    
    log(f"Step : {step+1}: model requested {len(tool_calls)} tool calls")

    messages.append({
      "role": "assistant",
      "content": full_response,
      "tool_calls": build_tool_calls_payload(tool_calls)
    })

    for tool_call in tool_calls:
      result = run_tool(tool_call)
      messages.append({
        "role": "tool",
        "tool_id": tool_call.id,
        "content": str(result),
      })
  else:
    # Ran out of steps: The model never gave the final answer
    fallback = "Sorry, I couldn't finish this within the step limit. "
    messages.append({
      "role": "assistant",
      "content": fallback
    })
    print("🤖Assistant: " + fallback)
    continue

    
 # Removed second LLM API call as loop is already added.
  
  messages.append({
    "role": "assistant",
    "content": full_response
  })


