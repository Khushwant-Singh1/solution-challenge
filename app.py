from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def query_handler(query: str):
    try:
        # Store query and process it with the model
        prompt = f"""
            You are a financial assistant helping the user with budgeting, investing, savings, or any general financial advice.
The user may ask for assistance with managing expenses, providing investment strategies, or answering general financial queries.
The conversation history is as follows:
            query = {query}
        """ 
        model = ChatOpenAI(model="gpt-3.5-turbo")
        with_message_history = RunnableWithMessageHistory(model, get_session_history)
        config = {"configurable": {"session_id": "abc2"}}
        
        # Invoke the model with the query
        response = with_message_history.invoke(
            [HumanMessage(content=prompt)],
            config=config,
        )
        
        # Print the model's response to the terminal
        print("Response:", response.content)
        
    except Exception as ex:
        print(f"Error: {ex}")

def main():
    print("Welcome to the Customer Service Assistant!")
    
    while True:
        # Take input from the user in terminal
        query = input("Please enter your query (or type 'exit' to quit): ")
        
        # Allow the user to exit the loop
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Process the query and get the response
        query_handler(query)

if __name__ == "__main__":
    main()
