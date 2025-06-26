# This script allows you to chat with the Gemini model and saves the conversation history to a JSON file.
import google.generativeai as genai
import json
import time
import os
import google.api_core.exceptions

# Configure Gemini API key
genai.configure(api_key="AIzaSyAp3PQ3sI-BgJZd5Yj03fPpFPVJCGlQSMU")

# Use the fast model to avoid quota issues
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Start chat session
chat = model.start_chat()

# Chat history to save
chat_history = []

# Load existing history if file exists
if os.path.exists("chat_history.json"):
    with open("chat_history.json", "r") as f:
        try:
            chat_history = json.load(f)
        except json.JSONDecodeError:
            chat_history = []

print("Chat started. Type 'stop' to end the chat.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "stop":
        print("Chat ended.")
        break
    try:
        response = chat.send_message(user_input)
        print("Gemini:", response.text)

        # Save this interaction
        chat_history.append({
            "user": user_input,
            "gemini": response.text
        })

        # Write chat history to JSON file
        with open("chat_history.json", "w") as f:
            json.dump(chat_history, f, indent=4)

    except google.api_core.exceptions.ResourceExhausted as e:
        print("Rate limit hit. Waiting 30 seconds...")
        time.sleep(30)
    except Exception as e:
        print("Error:", e)
