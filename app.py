import os
import time
import logging
import asyncio
import aiohttp
from openai import AsyncOpenAI
from googlesearch import search
from flask import Flask, render_template, request
from urllib.error import HTTPError
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.readers.web import SimpleWebPageReader
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-proj-REvgnhiXWATDv8GxI3kTT3BlbkFJNq4vknGlGqJg9ruKQNdl"

# Initialize the AsyncOpenAI client
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

async def google_search(query, num_results=2, delay=1):
    result_links = []
    try:
        logger.info(f"Performing Google search for query: {query}")
        for link in search(query, num=num_results, stop=num_results, pause=delay):
            result_links.append(link)
            logger.info(f"Found link: {link}")
            await asyncio.sleep(delay)
    except HTTPError as e:
        if e.code == 429:
            logger.warning(f"Rate limit exceeded for query: {query}. Waiting for 60 seconds.")
            await asyncio.sleep(1)
            return await google_search(query, num_results, 1)
        else:
            logger.error(f"HTTP Error {e.code}: {e.reason}")
    except Exception as e:
        logger.error(f"An error occurred during Google search: {str(e)}")
    return result_links

async def generate_additional_queries(query):
    logger.info(f"Generating additional queries for: {query}")
    prompt = f"Generate 5 variations of the following search query:\n\n\"{query}\""
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates search query variations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            temperature=0.7
        )
        additional_queries = response.choices[0].message.content.strip().split("\n")
        queries = [q.strip('- ').strip() for q in additional_queries if q.strip()]
        logger.info(f"Generated additional queries: {queries}")
        return queries
    except Exception as e:
        logger.error(f"Error generating additional queries: {str(e)}")
        return []

async def scrape_content(session, link):
    try:
        logger.info(f"Scraping content from: {link}")
        async with session.get(link, timeout=2) as response:
            if response.status == 200:
                html_content = await response.text()
                documents = SimpleWebPageReader(html_to_text=True).load_data([html_content])
                if documents and documents[0].text.strip():
                    content = documents[0].text.strip()
                    logger.info(f"Scraped content (truncated to 500 chars): {content[:500]}...")
                    return content
            logger.warning(f"No content found at: {link}")
    except Exception as e:
        logger.error(f"Error scraping {link}: {str(e)}")
    return ""

async def scrape_and_analyze(links, person_info):
    async with aiohttp.ClientSession() as session:
        tasks = [scrape_content(session, link) for link in links]
        contents = await asyncio.gather(*tasks)

    all_content = "\n\n".join(filter(None, contents))

    if not all_content.strip():
        logger.warning("No valid content found to analyze.")
        return "No valid content found to analyze."

    logger.info("Creating VectorStoreIndex from scraped content")
    index = VectorStoreIndex.from_documents([Document(text=all_content)])
    query_engine = index.as_query_engine()

    query = f"Based on the following information about a person: {person_info}, analyze if the content is discussing this person. Provide a detailed explanation."
    logger.info(f"Analyzing content with query: {query}")
    response = query_engine.query(query)

    logger.info(f"Analysis result: {str(response)}")
    return str(response)

def run_async(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        person_info = request.form['person_info']
        logger.info(f"Received search query: {query}")
        logger.info(f"Received person info: {person_info}")

        additional_queries = run_async(generate_additional_queries(query))
        all_links = []
        results = {}

        for additional_query in additional_queries:
            links = run_async(google_search(additional_query, num_results=3, delay=5))
            results[additional_query] = links
            all_links.extend(links)

        logger.info(f"Total links found: {len(all_links)}")
        analysis = run_async(scrape_and_analyze(all_links, person_info))
        return render_template('index.html', original_query=query, results=results, analysis=analysis)
    return render_template('index.html')

if __name__ == '__main__':
    logger.info("Starting the Flask application")
    app.run(debug=True)