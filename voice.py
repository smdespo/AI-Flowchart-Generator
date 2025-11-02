import os
from typing import Dict, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Initialize FastAPI app
app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Groq client
client = Groq(api_key=GROQ_API_KEY)

# ✅ Model for incoming data
class UserInput(BaseModel):
    R: int
    I: int
    A: int
    S: int
    E: int
    C: int
    conversation_id: str


# ✅ Conversation memory handler
class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that gives 5 career options based on RIASEC model. "
                    "You must only respond and list the 5 career options in a numbered format without any explanations or additional text."
                    "You Must only respond with the career names."
                    
                )
            }
        ]
        self.active: bool = True


# ✅ Conversation storage
conversations: Dict[str, Conversation] = {}


# ✅ Query Groq API
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


# ✅ Create or retrieve conversation
def get_or_create_conversation(conversation_id: str) -> Conversation:
    if conversation_id not in conversations:
        conversations[conversation_id] = Conversation()
    return conversations[conversation_id]


# ✅ Main chat route
@app.post("/chat/")
async def chat(input: UserInput):
    conversation = get_or_create_conversation(input.conversation_id)

    if not conversation.active:
        raise HTTPException(status_code=400, detail="Chat session ended")

    # Combine RIASEC scores
    riasec_summary = (
        f"R: {input.R}, I: {input.I}, A: {input.A}, "
        f"S: {input.S}, E: {input.E}, C: {input.C}"
    )

    # Instruction for Groq
    user_message = (
        f"Based on these RIASEC scores ({riasec_summary}), "
        f"suggest 5 suitable career options for the student. "
        f"Respond only with the 5 career names in numbered format."
    )

    conversation.messages.append({"role": "user", "content": user_message})

    response = query_groq_api(conversation)
    cleaned_response = response.strip()
    
   
    list_items = cleaned_response.split('\n')
    
    # Third, clean up each item (e.g., remove the number and period at the start of a line
    # if it accidentally got left by the model, or just strip extra space)
    processed_items = [item.strip() for item in list_items if item.strip()] # filter out empty strings
    
    
    final_response = '\n'.join(processed_items)
    final_response = final_response.replace("\\n", "\n")  # <-- add this line
    final_response = final_response.replace("\n", "\n")  # <-- add this line


    conversation.messages.append({"role": "assistant", "content": final_response})

    return {
        "response": final_response,
        "conversation_id": input.conversation_id
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
