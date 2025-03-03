import os
import datetime
import warnings
import google.generativeai as genai
from dotenv import load_dotenv
from termcolor import colored

# Suppress GRPC warnings
warnings.filterwarnings("ignore", category=UserWarning, module="grpc")

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

def get_daily_term():
    """Get today's financial term using Gemini with date-based seed"""
    today = datetime.date.today()
    seed = today.strftime("%Y%m%d")
    
    prompt = f"""Generate one random financial term suitable for daily learning. 
    The term should be different based on this seed: {seed}.
    Return only the term itself, nothing else."""
    
    try:
        response = model.generate_content(prompt, request_options={"timeout": 10})
        return response.text.strip()
    except Exception as e:
        return "Unearned Revenue"  # Fallback term

def get_term_explanation(term):
    """Get explanation and product example for the given term"""
    prompt = f"""Explain the financial term '{term}' in simple terms with these components:
    1. Brief definition (1-2 sentences)
    2. Common usage context
    3. Example financial product that uses this concept
    4. Simple real-world analogy
    5. Key considerations for consumers
    
    Format the response with clear section headings between each component, 
    but do not use markdown formatting. Use '---' as section separators."""
    
    try:
        response = model.generate_content(prompt, request_options={"timeout": 10})
        return response.text
    except Exception as e:
        return f"""Brief Definition:
Unearned revenue represents money received for undelivered goods/services.
---        
Common Usage Context:
Common in subscriptions, software, and service industries.
---        
Example Financial Product:
Annual software subscriptions
---        
Real-World Analogy:
Like a gift certificate for future spa services
---        
Key Considerations:
Check company stability before large upfront payments"""

def display_explanation(term, explanation):
    """Format and display the explanation with colors"""
    sections = explanation.split('---')
    
    print(colored(f"\n📈 Daily Financial Term: {term}", "cyan", attrs=["bold"]))
    print(colored("=" * 50, "blue"))
    
    for section in sections:
        lines = section.strip().split('\n')
        title = lines[0].strip()
        content = '\n'.join(lines[1:]).strip()
        
        print(colored(f"\n✨ {title}", "yellow"))
        print(colored(content, "white"))
        
    print(colored("\n💡 Remember: Financial literacy is a superpower!", "magenta"))

def main():
    try:
        term = get_daily_term()
        explanation = get_term_explanation(term)
        display_explanation(term, explanation)
    except Exception as e:
        print(colored(f"⚠️ Error: {str(e)}", "red"))
    finally:
        # Graceful shutdown with error suppression
        try:
            if hasattr(genai, '_client_manager'):
                genai._client_manager._client.close()
        except:
            pass  # Ignore any cleanup errors


        
if __name__ == "__main__":
    print(colored("\n🔍 Your Daily Financial Education Briefing", "green", attrs=["bold"]))
    main()