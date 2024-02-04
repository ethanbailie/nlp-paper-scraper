import psycopg2
import os
from openai import OpenAI
import pinecone
from dotenv import load_dotenv

## the job of this class is to provide functions to classify the text data
## the idea is to do a semantic search across the embedddings of the abstracts by searching for the categories
## take the highest semantic scores and assign the category to the paper
## remake the index with the resulting information
class classify:
    def __init__(self, user='postgres', host='localhost'):
        ## get env variables
        load_dotenv()
        
        ## set up postgres connector
        self.user = user
        self.host = host
        self.password = os.getenv('postgresPass')
        self.database = 'postgres'

        openai_key=os.getenv('openaiKey')
        pinecone_key=os.getenv('pineconeKey')

        conn = psycopg2.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password
        )

        self.cur = conn.cursor()

        ## instantiate variables
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.embed_model = "text-embedding-ada-002"
        self.env = "gcp-starter"
        self.index_name = 'nlp-embedding'

        ## initialize openai client
        self.openAIClient = OpenAI(
            api_key=self.openai_key
        )

        ## initialize pinecone
        pinecone.init(api_key=self.pinecone_key, environment=self.env)
        self.index = pinecone.Index(self.index_name)

    def semanticSearch(self, uxFilters, searchInput=None, topKInput=100):
        ## embed search query
        res = self.openAIClient.embeddings.create(
            input=searchInput,
            model=self.embed_model
        )
        searchEmbedding = res.data[0].embedding

        ## query Pinecone
        count = 0
        while count < 3:
            try:
                searchResult = self.index.query(
                    vector=searchEmbedding, 
                    filter=uxFilters,
                    top_k=topKInput,
                    include_metadata = True
                )
                break
            except Exception as e:
                print(e)
                print("Search failed.")
                count +=1

        matches = searchResult['matches']
        return matches
    
    

