import requests
import json

# Replace with your actual Gemini API key
API_KEY = "AIzaSyAp3PQ3sI-BgJZd5Yj03fPpFPVJCGlQSMU"

# Endpoint URL for Gemini API
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# HTTP headers
headers = {
    "Content-Type": "application/json"
}

print("Chat started with Gemini! Type 'stop' to end the chat.\n")

# Chat loop
while True:
    user_input = input("You: ")

    if user_input.lower() == "stop":
        print("Chat ended.")
        break

    # Prepare data for POST request
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": user_input
                    }
                ]
            }
        ]
    }

    # Send request to Gemini
    response = requests.post(url, headers=headers, data=json.dumps(data))

    try:
        # Parse and print the model's response
        gemini_reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        print("Gemini:", gemini_reply)
    except Exception as e:
        print("Something went wrong:", response.text)
