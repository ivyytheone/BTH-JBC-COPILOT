import asyncio
import os
from dataclasses import dataclass
from typing import List, Optional
#Unsure, but i think this .py file is redundent since we use live indexed data and arent even running this scriptfile.
#adding comments anyway.

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    ComplexField,
    CorsOptions,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)
from teams.ai.embeddings import AzureOpenAIEmbeddings, AzureOpenAIEmbeddingsOptions

from get_data import get_doc_data

from dotenv import load_dotenv

load_dotenv(f'{os.getcwd()}/env/.env.testtool.user')

@dataclass
#Setup dataclasses that are existing in the index
class Doc:
    chunk_id: Optional[str] = None
    title: Optional[str] = None
    #description: Optional[str] = None
    text_vector: Optional[List[float]] = None

async def upsert_documents(client: SearchClient, documents: list[Doc]):
    return client.merge_or_upload_documents(documents)

#add the index-fields that are existing in index.
async def create_index_if_not_exists(client: SearchIndexClient, name: str):
    doc_index = SearchIndex(
        name=name,
        fields = [
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="title", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="text_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name='vector-jbc-testdata-algorithm'),
        ],
        scoring_profiles=[],
        cors_options=CorsOptions(allowed_origins=["*"]),
        vector_search = VectorSearch(
            profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-algorithms-config")],
            algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")],
        )
    )

    client.create_or_update_index(doc_index)

async def setup(search_api_key, search_api_endpoint):
    index = 'vector-jbc-testdata'

    credentials = AzureKeyCredential(search_api_key)

    search_index_client = SearchIndexClient(search_api_endpoint, credentials)
    await create_index_if_not_exists(search_index_client, index)
    
    print("Create index succeeded. If it does not exist, wait for 5 seconds...")
    await asyncio.sleep(5)

    search_client = SearchClient(index, credentials)

    embeddings = AzureOpenAIEmbeddings(AzureOpenAIEmbeddingsOptions(
        azure_api_key=os.getenv('SECRET_AZURE_OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        azure_deployment=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')
    ))
    data = await get_doc_data(embeddings=embeddings)
    await upsert_documents(search_client, data)

    print("Upload new documents succeeded. If they do not exist, wait for several seconds...")
    
search_api_key = os.getenv('SECRET_AZURE_SEARCH_KEY')
search_api_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
asyncio.run(setup(search_api_key, search_api_endpoint))
print("setup finished")

