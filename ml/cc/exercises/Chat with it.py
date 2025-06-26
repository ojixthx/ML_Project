import google.generativeai as genai

# Set up your Gemini API key
genai.configure(api_key="AIzaSyAp3PQ3sI-BgJZd5Yj03fPpFPVJCGlQSMU")
# AIzaSyAp3PQ3sI-BgJZd5Yj03fPpFPVJCGlQSMU

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-pro')

# Start a chat session
chat = model.start_chat()

print("Chat started. Type 'stop' to end the chat.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "stop":
        print("Chat ended.")
        break
    response = chat.send_message(user_input)
    print("Gemini:", response.text)
