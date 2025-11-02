import os
from typing import Dict, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
GROQ_API_KEY = "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"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

client = Groq(api_key=GROQ_API_KEY)


class UserInput(BaseModel):
    field: str = ""
    message: str = ""
    role: str = "user"
    conversation_id: str


class Conversation:
    def __init__(self, field: str):
    
        self.messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that answers questions related to the career option chosen: {field}. "
                    "Keep answers short — maximum 1-2 sentences. Be clear and factual."
                    "Your responses must be limited to a maximum of one or two sentences.** Do not use bullet points or numbered lists."

                )
            }
        ]
        self.active: bool = True


conversations: Dict[str, Conversation] = {}


def query_groq_api(conversation: Conversation) -> str:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversation.messages,
            temperature=0.7,
            top_p=1,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with Groq API: {str(e)}")


def get_or_create_conversation(conversation_id: str, field: str) -> Conversation:
    if conversation_id not in conversations:
        conversations[conversation_id] = Conversation(field)
    return conversations[conversation_id]


@app.post("/chat/")
async def chat(input: UserInput):
    conversation = get_or_create_conversation(input.conversation_id, input.field)

    if not conversation.active:
        raise HTTPException(status_code=400, detail="Chat session ended")

    try:
        conversation.messages.append({"role": input.role, "content": input.message})

        response = query_groq_api(conversation)
        response = response.strip()
        conversation.messages.append({"role": "assistant", "content": response})

        return {
            "response": response,
            "conversation_id": input.conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chat session ended: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
