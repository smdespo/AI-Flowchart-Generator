import os
from typing import Dict, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("API key for Groq is missing. Please set in dotenv file")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

client = Groq(api_key=GROQ_API_KEY)

class userinput(BaseModel):
    message: str=""
    role: str = "user"
    conversation_id: str


class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that generates flowcharts. "
" For any topic given by the user, you must respond with ONLY the Mermaid.js code for an attractive and well-structured flowchart."
"Do NOT include any explanations, greetings, or additional text"
"The response should start with `graph TD` or `flowchart LR` and contain only the necessary syntax"
" Use appropriate Mermaid.js elements like nodes, edges, subgraphs, and styles to enhance. "
"For example, if the user describes a login process"
"your response should be just this"
"\n\ngraph TD\n    A[User enters credentials] --> B{Valid?}\n    B -->|Yes| C[Access granted]\n    B -->|No| A"

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
            temperature=1,
            top_p=1,
            stream=False   
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with Groq API: {str(e)}")


    


def get_or_createconversation(conversation_id: str) -> Conversation:
    if conversation_id not in conversations:
        conversations[conversation_id] = Conversation() 
    return conversations[conversation_id]


@app.post("/chat/")
async def chat(input: userinput):
    conversation = get_or_createconversation(input.conversation_id)

    if not conversation.active:
        raise HTTPException(status_code=400, detail="Chat session ended")

    try:
        conversation.messages.append(  
            {"role": input.role, "content": input.message}
        )

        response = query_groq_api(conversation)
        mermaid_code = response.strip()
        conversation.messages.append({"role": "assistant", "content": mermaid_code})
      
        
        return {
            "mermaid": mermaid_code,
            "conversation_id": input.conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chat session ended: {str(e)}")


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)