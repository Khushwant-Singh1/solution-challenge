from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from gtts import gTTS
import os
from typing import List, Dict
import asyncio
from dotenv import load_dotenv
import feedparser
import httpx
from readability import Document
import subprocess

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_RSS_URL = "https://finance.yahoo.com/news/rssindex"
TIMEOUT = 25
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

class FinancialNewsAnchor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY
        )
        self.summary_prompt = self._create_summary_prompt()
    
    def _create_summary_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            """Analyze this financial news article. If content is unavailable, provide a cautious summary based on the title alone:
            
            Title: {title}
            Content: {content}
            
            Concise Summary (max 100 words):"""
        )

    async def fetch_news(self) -> List[Dict]:
        """Fetch news articles with improved filtering"""
        try:
            feed = feedparser.parse(NEWS_RSS_URL)
            return [
                {
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", "")
                } 
                for entry in feed.entries[:5] if entry.get("link")
            ]
        except Exception as e:
            print(f"Error fetching RSS feed: {e}")
            return []

    async def fetch_article_content(self, url: str) -> str:
        """Fetch and parse article content with Readability"""
        async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                
                # Extract main content using Readability
                doc = Document(response.text)
                content = doc.summary()
                return content if len(content) > 100 else "Content unavailable"
                
            except Exception as e:
                print(f"Error processing {url}: {e}")
                return ""

    async def summarize_articles(self, articles: List[Dict]) -> List[str]:
        """Generate reliable summaries with fallback"""
        chain = (
            {"title": RunnablePassthrough(), "content": RunnablePassthrough()}
            | self.summary_prompt
            | self.llm
        )
        
        summaries = []
        for article in articles:
            try:
                content = await self.fetch_article_content(article["link"])
                result = await chain.ainvoke({
                    "title": article["title"],
                    "content": content[:5000]  # Limit for API constraints
                })
                summaries.append(result.content.strip())
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Error summarizing {article['title']}: {e}")
                summaries.append(f"Update: {article['title']}")
        
        return summaries or ["Important financial updates are currently unavailable"]

    def text_to_speech(self, text: str) -> str:
        """Robust TTS with validation"""
        if not text.strip():
            text = "No financial updates available"
            
        try:
            tts = gTTS(text=text, lang='en', tld='co.uk', slow=False)
            audio_file = "news_summary.mp3"
            tts.save(audio_file)
            return audio_file
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            return None

    def play_audio(self, filename: str):
        """Reliable Linux audio playback"""
        if not os.path.exists(filename):
            return
            
        try:
            # Try mpv first, then fallback to paplay
            subprocess.run(["mpv", "--really-quiet", filename], check=True)
        except:
            try:
                subprocess.run(["paplay", filename], check=True)
            except Exception as e:
                print(f"Audio playback failed: {e}")

async def main():
    anchor = FinancialNewsAnchor()
    
    print("Fetching latest financial news...")
    articles = await anchor.fetch_news()
    
    if not articles:
        print("No articles found")
        return
    
    print("Analyzing news articles...")
    summaries = await anchor.summarize_articles(articles)
    
    broadcast = "\n\n".join([
        f"News Update {i+1}: {summary}" 
        for i, summary in enumerate(summaries) if summary
    ])
    
    print("\nToday's Financial Headlines:")
    print(broadcast or "No summaries generated")
    
    print("\nGenerating audio broadcast...")
    audio_file = anchor.text_to_speech(broadcast)
    
    if audio_file:
        anchor.play_audio(audio_file)
        os.remove(audio_file)

if __name__ == "__main__":
    asyncio.run(main())