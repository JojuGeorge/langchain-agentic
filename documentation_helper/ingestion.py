import asyncio
import os
import ssl
import certifi
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeSparseVectorStore
# from langchain_chroma import Chroma
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import (Colors, log_error, log_header,
                    log_info, log_success, log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embedding_model = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=os.environ.get('OPENAI_API_KEY'),
    show_progress_bar=True,
    chunk_size=50,
    retry_min_seconds=10
)

# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
vectore_store = PineconeSparseVectorStore(
    index_name=os.environ['INDEX_NAME'], embedding=embedding_model)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # Crawl the documentation site

    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 1,
        "extract_depth": "advanced"
    })

    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(Document(
            page_content=tavily_crawl_result_item["raw_content"],
            metadata={"source": tavily_crawl_result_item["url"]}
        ))

     # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )


if __name__ == "__main__":
    asyncio.run(main())
