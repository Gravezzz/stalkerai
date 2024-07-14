import os
import time
import logging
import asyncio
from openai import AsyncOpenAI
from googlesearch import search
from flask import Flask, render_template, request
from urllib.error import HTTPError
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.readers.web import SimpleWebPageReader
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the AsyncOpenAI client
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

async def google_search(query, num_results):
    result_links = []
    try:
        logger.info(f"Performing Google search for query: {query}")
        for link in search(query, num=num_results, stop=num_results):
            if link.startswith('http://') or link.startswith('https://'):
                result_links.append(link)
                logger.info(f"Found link: {link}")
            else:
                logger.warning(f"Skipping invalid link: {link}")
    except HTTPError as e:
        if e.code == 429:
            logger.warning(f"Rate limit exceeded for query: {query}. Waiting for 60 seconds.")
            await asyncio.sleep(60)
            return await google_search(query, num_results)
        else:
            logger.error(f"HTTP Error {e.code}: {e.reason}")
    except Exception as e:
        logger.error(f"An error occurred during Google search: {str(e)}")
    return result_links

async def generate_additional_queries(full_name, short_bio, social_media, research_goal):
    logger.info(f"Generating additional queries for: {full_name}, {short_bio}, {social_media}, {research_goal}")
    prompt = (
        f"Using the following information:\n"
        f"Full Name: {full_name}\n"
        f"Short Bio: {short_bio}\n"
        f"Social Media: {social_media}\n"
        f"Research Goal: {research_goal}\n\n"
        f"Generate 3 variations of a search query to find relevant information about this person."
    )
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
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
        logger.info(f"Generated additional queries: {queries[:3]}")
        return queries[:3]  # Return top 3 queries
    except Exception as e:
        logger.error(f"Error generating additional queries: {str(e)}")
        return []

async def scrape_content(session, link):
    try:
        logger.info(f"Scraping content from: {link}")
        async with session.get(link, timeout=10) as response:
            if response.status == 200 and 'text/html' in response.headers['Content-Type']:
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                # Extract the text from the HTML content
                text_content = ' '.join(soup.stripped_strings)
                if text_content:
                    logger.info(f"Scraped content (truncated to 500 chars): {text_content[:500]}...")
                    return text_content
                logger.warning(f"No valid content found at: {link}")
            else:
                logger.warning(f"Failed to retrieve HTML content from: {link}, Status code: {response.status}, Content-Type: {response.headers['Content-Type']}")
    except aiohttp.ClientError as e:
        logger.error(f"Client error scraping {link}: {str(e)}")
    except Exception as e:
        logger.error(f"Error scraping {link}: {str(e)}")
    return ""

async def scrape_and_analyze(links, person_info, full_name, research_goal):
    async with aiohttp.ClientSession() as session:
        tasks = [scrape_content(session, link) for link in links]
        contents = await asyncio.gather(*tasks)

    all_content = "\n\n".join(filter(None, contents))

    if not all_content.strip():
        logger.warning("No valid content found to analyze.")
        return "No valid content found to analyze.", ""

    logger.info("Creating VectorStoreIndex from scraped content")
    index = VectorStoreIndex.from_documents([Document(text=all_content)])
    query_engine = index.as_query_engine()

    query = f"""You are a expert marketing agent with several decades of experience. You must use the information below to craft a highly personalised message to the target. 
    The message MUST be written clearly and referencing specific knowledge that is added below about the targets recent activity.
    The message MUST act as a hook to get the target to reply. The target is {full_name} for the purpose of {research_goal}. Use this information: {person_info}"""
    logger.info(f"Analyzing content with query: {query}")
    response = query_engine.query(query)

    logger.info(f"Analysis result: {str(response)}")
    return str(response), all_content[:2000]  # Limit to 2000 characters to avoid too much content

def run_async(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        full_name = request.form['full_name']
        short_bio = request.form['short_bio']
        social_media = request.form['social_media']
        research_goal = request.form['research_goal']
        logger.info(f"Received full name: {full_name}")
        logger.info(f"Received short bio: {short_bio}")
        logger.info(f"Received social media link: {social_media}")
        logger.info(f"Received research goal: {research_goal}")

        additional_queries = run_async(generate_additional_queries(full_name, short_bio, social_media, research_goal))
        all_links = []
        results = {}

        for additional_query in additional_queries:
            links = run_async(google_search(additional_query, num_results=3))
            results[additional_query] = links
            all_links.extend(links)

        logger.info(f"Total links found: {len(all_links)}")
        analysis, scraped_content = run_async(scrape_and_analyze(all_links, research_goal, full_name, research_goal))
        return render_template('index.html', original_query=full_name, results=results, analysis=analysis, scraped_content=scraped_content)
    return render_template('index.html')

if __name__ == '__main__':
    logger.info("Starting the Flask application")
    app.run(debug=True)
