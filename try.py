from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()

DB_FAISS_PATH = 'vectorstore/db_faiss'
store = {}

FINANCIAL_SYSTEM_PROMPT = """You are a financial guidance assistant for Indian investors. Your responses must:
1. Be accurate and based on SEBI regulations
2. Explain concepts simply (avoid jargon)
3. Highlight risks for investment products
4. Disclaim "I am not a SEBI-certified advisor"
5. Prioritize information from the context"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", FINANCIAL_SYSTEM_PROMPT),
    ("human", """Chat History:
{chat_history}

Relevant Context:
{context}

User Profile:
- Risk Tolerance: {risk_tolerance}
- Investment Goal: {investment_goal}
- Experience Level: {experience_level}

Question: {question}""")
])

def load_llm():
    return ChatOpenAI(model="gpt-4", temperature=0.3)

def load_vectorstore():
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

class FinancialAssistant:
    def __init__(self):
        self.db = load_vectorstore()
        self.llm = load_llm()
        
        # Updated safety check using new chain syntax
        self.safety_check = (
            PromptTemplate.from_template("Verify compliance with Indian financial regulations: {response}")
            | ChatOpenAI(model="gpt-3.5-turbo")
        )
        
    def get_memory(self, session_id):
        if session_id not in store:
            store[session_id] = ConversationBufferMemory(
                return_messages=True,
                output_key="answer",
                memory_key="chat_history"
            )
        return store[session_id]
    
    def get_user_profile(self, session_id):
        return {
            'risk_tolerance': 'moderate',
            'investment_goal': 'retirement',
            'experience_level': 'beginner'
        }
    
    def query(self, session_id, question):
        memory = self.get_memory(session_id)
        user_profile = self.get_user_profile(session_id)
        
        chain = (
            RunnablePassthrough.assign(
                chat_history=lambda _: memory.load_memory_variables({})["chat_history"],
                context=lambda x: self.db.as_retriever().invoke(x["question"])
            )
            | PROMPT_TEMPLATE
            | self.llm
        )
        
        response = chain.invoke({
            "question": question,
            **user_profile
        })
        
        # Save to memory
        memory.save_context(
            {"question": question},
            {"answer": response.content}
        )
        
        # Safety check
        safety_result = self.safety_check.invoke({"response": response.content})
        if "non-compliant" in safety_result.content.lower():
            return "I cannot provide that information. Please consult a certified financial advisor."
        
        return response.content

if __name__ == "__main__":
    print("Namaste! I'm your financial guide. Ask me about:")
    print("- Basic investing concepts\n- Indian market products\n- Financial planning\nType 'exit' to quit.")
    
    assistant = FinancialAssistant()
    session_id = "demo_user"
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                print("Dhanyavaad! Always verify investments with certified advisors.")
                break
                
            response = assistant.query(session_id, user_input)
            print(f"Guide: {response}")
            
        except Exception as e:
            print(f"System error: {str(e)}")
            break