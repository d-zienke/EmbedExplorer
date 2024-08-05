import logging
from chatbot.response_generator import ResponseGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    response_generator = ResponseGenerator()

    print("Chatbot is ready to converse! Type your query below.")
    while True:
        query_text = input("You: ")
        if query_text.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        response = response_generator.decide_response(query_text)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
