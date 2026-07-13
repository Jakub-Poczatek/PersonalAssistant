import json
import logging
from ollama import chat
from ollama import ChatResponse

logger = logging.getLogger("assistant.nlu")

MODEL = "phi3"
NLU_PROMPT = """
You are an intent classifier for a voice assistant.
Given a voice command, return ONLY a JSON object with:
- skill: the name of the skill to run
- params: any parameters needed

Available skills:
- "time.get_time": returns the current time. No params needed.
- "time.get_date": returns today's day and date. No params needed.
- "weather.get_weather": returns the weather today. No params needed.
- "default.unknown": use this if no skills match.

Return ONLY valid JSON. No explanation. No other text.

Example outputs:
{"skill": "time.get_time", "params": {}}
{"skill": "weather.get_weather", "params": {}}
{"skill": "default.unknown", "params": {}}
"""

def nlu(command: str) -> dict:
    logger.debug(command)
    message = [
        {
            "role": "user",
            "content": f"{NLU_PROMPT} \n{command}"
        },
    ]
    chat_response: ChatResponse = chat(model=MODEL, messages=message)
    response = str(chat_response.message.content)

    response = response.strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[1] # remove first line (```json)
        response = response.rsplit("```", 1)[0] # remove trailing ```
        response = response.strip()
    try:
        response_json = json.loads(response)
    except json.JSONDecodeError:
        return {"skill": "default.unknown", "params": {}}
    
    return {
        "skill": response_json.get("skill", "default.unknown"),
        "params": response_json.get("params", {})
    }
