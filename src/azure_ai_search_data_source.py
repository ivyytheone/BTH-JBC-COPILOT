from dataclasses import dataclass
from typing import Optional, List
from azure.search.documents.indexes.models import _edm as EDM
from azure.search.documents.models import VectorQuery, VectorizedQuery
from teams.ai.embeddings import AzureOpenAIEmbeddings, AzureOpenAIEmbeddingsOptions
from teams.state.memory import Memory
from teams.state.state import TurnContext
from teams.ai.tokenizers import Tokenizer
from teams.ai.data_sources import DataSource

from config import Config
#Fetching Index data from Azure Ai Search.
#Data in Azurea Ai Search is setup with an Indexer that index data on a Azure StorageAccount.
async def get_embedding_vector(text: str):
    embeddings = AzureOpenAIEmbeddings(AzureOpenAIEmbeddingsOptions(
        azure_api_key=Config.AZURE_OPENAI_API_KEY,
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        azure_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    ))
    
    result = await embeddings.create_embeddings(text)
    if (result.status != 'success' or not result.output):
        raise Exception(f"Failed to generate embeddings for description: {text}")
    
    return result.output[0]

@dataclass
#Setup dataclasses that are existing in the index
class Doc:
    chunk_id: Optional[str] = None
    parent_id: Optional[str] = None
    chunk: Optional[str] = None
    title: Optional[str] = None
    #description: Optional[str] = None
    text_vector: Optional[List[float]] = None

@dataclass
class AzureAISearchDataSourceOptions:
    name: str
    indexName: str
    azureAISearchApiKey: str
    azureAISearchEndpoint: str

#Add packets for AzureSearch and AzureIndexing
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import json

@dataclass
class Result:
    def __init__(self, output, length, too_long):
        self.output = output
        self.length = length
        self.too_long = too_long

class AzureAISearchDataSource(DataSource):
    def __init__(self, options: AzureAISearchDataSourceOptions):
        self.name = options.name
        self.options = options
        self.searchClient = SearchClient(
            options.azureAISearchEndpoint,
            options.indexName,
            AzureKeyCredential(options.azureAISearchApiKey)
        )
        
    def name(self):
        return self.name

    async def render_data(self, _context: TurnContext, memory: Memory, tokenizer: Tokenizer, maxTokens: int):
        query = memory.get('temp.input')
        embedding = await get_embedding_vector(query)
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=2, fields="text_vector")

        if not query:
            return Result('', 0, False)
        #add the index-fields that are existing in index.
        selectedFields = [
            'chunk_id',
            'parent_id',
            'chunk',
            'title',
            'text_vector',
        ]

        searchResults = self.searchClient.search(
            search_text=query,
            select=selectedFields,
            vector_queries=[vector_query],
        )

        if not searchResults:
            return Result('', 0, False)

        usedTokens = 0
        doc = ''
        for result in searchResults:
            source_info = f"Source: {result['title']}\n" #This could be updated to retrive a card for previewing the DATA/PDF.
            tokens = len(tokenizer.encode(json.dumps(result["title"]) + source_info))

            if usedTokens + tokens > maxTokens:
                break
            doc += source_info
            doc += json.dumps(result["chunk"]) + "\n\n" #This return the chunk/text of indexed data.
            usedTokens += tokens

        return Result(doc, usedTokens, usedTokens > maxTokens)