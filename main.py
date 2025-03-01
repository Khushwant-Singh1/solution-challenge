from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()
# Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  

DB_FAISS_PATH = 'vectorstore/db_faiss'

# System prompt for guiding the AI model
SYSTEM_PROMPT = "Please provide a helpful answer based on the context and question provided."
custom_prompt_template = f"""
{SYSTEM_PROMPT}  # Hardcoded system prompt

Context: {{context}}
Question: {{question}}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Loading OpenAI model
def load_llm():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
    return llm

# QA Model Function
def qa_bot():
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    print("Required input keys:", qa_result.input_keys)
    response = qa_result.invoke({
        
        "query": query
    })
    return response

if __name__ == "__main__":
    print("Welcome to the CLI-based QA Bot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        result = final_result(user_input)
        print("Bot:", result["result"])
