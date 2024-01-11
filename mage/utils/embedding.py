import psycopg2
import os
from openai import OpenAI
import pinecone
from dotenv import load_dotenv

class embedder:
    def __init__(self, user='postgres', password=None, openai=None, pinecone=None):
        load_dotenv()
        
        self.user = user
        self.password = os.getenv('postgresPass')
        self.database = 'postgres'

        conn = psycopg2.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password
        )

        self.cur = conn.cursor()

        self.openai = openai
        self.pinecone = pinecone
        self.embed_model = "text-embedding-ada-002"
        self.env = "gcp-starter"
        self.index_name = 'nlp-embedding'

        self.openAIClient = OpenAI(
            api_key=self.openai
        )

        pinecone.init(api_key=self.pinecone, environment=self.env)
        self.index = pinecone.Index(self.index_name)

    def createIndex(self):
        ## Check if index already exists, create if not.
        if len(pinecone.list_indexes()) == 0 or pinecone.list_indexes()[0] != self.index_name:
            pinecone.create_index(
                self.index_name,
                dimension=1536,  # dimensionality of text-embedding-ada-002
                metric='cosine',
            )
            
        # view index stats
        self.index.describe_index_stats()

    def paperQuery(self):
        papers = self.cur.execute(
            """
            SELECT *
            FROM public.raw_papers
            ORDER BY updated DESC
            """
        )

        papers.columns = [
            'id',
            'title',
            'summary',
            'published',
            'updated'
        ]

        return papers