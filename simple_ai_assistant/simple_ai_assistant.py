# standard library imports
import os

# third party imports
from openai import OpenAI
from dotenv import load_dotenv

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
    "content": "Act like a very humble agent and respond always under 20 words."
  }
)

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
    response = client.chat.completions.create(
      model=os.getenv("LLM_MODEL"),
      messages=messages,
      tools=TOOLS
    )
    msg = response.choices[0].message
    messages.append(msg)

    if not msg.tool_calls: 
      break
    
    log(f"Step : {step+1}: model requested {len(msg.tool_calls)} tool calls")

    for tool_call in msg.tool_calls:
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
    "content": msg.content
  })

  print("🤖Asisstant: " +(msg.content or "")) #response from the LLM

