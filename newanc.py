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
import random
import time

load_dotenv()

# Configuration for Indian financial markets
INDIAN_NEWS_RSS = [
    "https://www.moneycontrol.com/rss/business.xml",
    "https://www.livemint.com/rss/money",
    "https://www.business-standard.com/rss/markets-101.rss"
]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
]
HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.google.co.in/"
}

class IndiaMarketAnchor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.summary_prompt = self._create_indian_prompt()
    
    def _create_indian_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            """Analyze this Indian market news for retail investors:
            
            Title: {title}
            Content: {content}
            
            Provide 1-sentence summary focusing on:
            - Market impact (Nifty/Sensex)
            - Key sector/company
            - Investor action
            
            If content unavailable, create title-based summary (12 words max)
            
            Summary:"""
        )

    def _get_source(self, rss_url: str) -> str:
        """Map RSS URL to source name"""
        if "moneycontrol" in rss_url:
            return "MoneyControl"
        elif "livemint" in rss_url:
            return "Livemint"
        elif "business-standard" in rss_url:
            return "Business Standard"
        return "Financial News"

    async def fetch_news(self) -> List[Dict]:
        """Fetch and validate news from Indian sources"""
        articles = []
        for rss_url in INDIAN_NEWS_RSS:
            try:
                print(f"Fetching {rss_url}...")
                feed = feedparser.parse(rss_url)
                print(f"Found {len(feed.entries)} entries")
                
                for entry in feed.entries[:3]:  # Get top 3 entries
                    if not hasattr(entry, 'link') or not hasattr(entry, 'title'):
                        continue
                    articles.append({
                        "title": entry.title,
                        "link": entry.link,
                        "source": self._get_source(rss_url),
                        "timestamp": time.time()
                    })
            except Exception as e:
                print(f"RSS Error {rss_url}: {str(e)}")
        return sorted(articles, key=lambda x: x["timestamp"])[:5]

    async def fetch_article_content(self, url: str) -> str:
        """Fetch content with Indian website compatibility"""
        headers = {**HEADERS, "User-Agent": random.choice(USER_AGENTS)}
        
        async with httpx.AsyncClient(
            timeout=30,
            headers=headers,
            follow_redirects=True,
            http2=True
        ) as client:
            try:
                response = await client.get(url)
                if response.status_code != 200:
                    return f"http_error_{response.status_code}"
                
                doc = Document(response.text)
                content = doc.summary()
                return content if 100 < len(content) < 10000 else "content_unavailable"
                
            except Exception as e:
                print(f"Fetch Error {url}: {str(e)}")
                return "error"

    async def analyze_articles(self, articles: List[Dict]) -> List[str]:
        """Generate market analysis with error handling"""
        chain = (
            {"title": RunnablePassthrough(), "content": RunnablePassthrough()}
            | self.summary_prompt
            | self.llm
        )
        
        analyses = []
        for article in articles:
            try:
                content = await self.fetch_article_content(article["link"])
                if any(err in content for err in ["error", "http_error", "unavailable"]):
                    analyses.append(self._title_based_summary(article))
                    continue
                
                result = await chain.ainvoke({
                    "title": article["title"],
                    "content": content[:2000]  # Conservative limit
                })
                analyses.append(f"{article['source']}: {result.content.strip()}")
                await asyncio.sleep(random.uniform(1, 3))
                
            except Exception as e:
                print(f"Analysis Error: {str(e)}")
                analyses.append(self._title_based_summary(article))
        
        return analyses or ["No market updates available"]

    def _title_based_summary(self, article: Dict) -> str:
        """Generate fallback summary from title"""
        return f"{article['source']}: {article['title'][:120]} (Full article unavailable)"

    def generate_broadcast(self, analyses: List[str]) -> str:
        """Format market briefing"""
        return "\n".join([
            f"📈 {analysis}" 
            for analysis in analyses 
            if analysis.strip()
        ]) + "\n\nEnd of Indian market update"

    def text_to_speech(self, text: str) -> str:
        """Convert to Indian-accent audio"""
        try:
            tts = gTTS(
                text=text,
                lang='en',
                tld='co.in',
                slow=False,
                lang_check=False
            )
            audio_file = "indian_market.mp3"
            tts.save(audio_file)
            return audio_file
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            return None

    def play_audio(self, filename: str):
        """Robust audio playback"""
        if not os.path.exists(filename):
            return
            
        try:
            subprocess.run(
                ["mpv", "--really-quiet", filename],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except:
            try:
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", filename],
                    check=True
                )
            except Exception as e:
                print(f"Playback Failed: {str(e)}")

async def main():
    anchor = IndiaMarketAnchor()
    
    print("🕵️ Fetching Indian market news...")
    articles = await anchor.fetch_news()
    print(f"📰 Found {len(articles)} articles")
    
    print("🔍 Analyzing for investor impact...")
    analyses = await anchor.analyze_articles(articles)
    
    broadcast = anchor.generate_broadcast(analyses)
    print("\n🇮🇳 Indian Market Briefing:")
    print(broadcast)
    
    print("\n🔊 Generating audio update...")
    if audio_file := anchor.text_to_speech(broadcast):
        print("▶️ Playing market update...")
        anchor.play_audio(audio_file)
        os.remove(audio_file)
        print("✅ Playback completed")

if __name__ == "__main__":
    asyncio.run(main())