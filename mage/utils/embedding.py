import psycopg2
import os
import time
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import pandas as pd

class embedder:
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

        # instantiate variables
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.embed_model = "text-embedding-ada-002"
        self.index_name = 'nlp-embedding'

        ## initialize openai client
        self.openAIClient = OpenAI(
            api_key=self.openai_key
        )

        ## initialize pinecone
        self.pc = Pinecone(api_key=self.pinecone_key)
        self.index = self.pc.Index(self.index_name)

    def createIndex(self):
        ## check if index already exists, create if not
        exists = False
        for index in self.pc.list_indexes():
            if self.index_name == index['name']:
                exists = True

        if not exists:
            print(self.pc.list_indexes()[0]['name'])
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # dimensionality of embedding model
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
        ## if the index exists, delete and recreate
        else:
            self.pc.delete_index(self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # dimensionality of embedding model
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
        ## view index stats
        self.index.describe_index_stats()

    def paperQuery(self):
        ## grabs all papers in db
        self.cur.execute(
            """
            SELECT *
            FROM public.raw_papers
            ORDER BY updated DESC
            """
        )

        papers = pd.DataFrame(self.cur.fetchall())

        papers.columns = [
            'id',
            'title',
            'summary',
            'published',
            'updated'
        ]
        return papers
    
    def embed(self, papers):
        ## intantiate variables
        upsert_array = []
        max_retries = 3
        wait_period = 5

        total_papers = len(papers)
        print(f"Total papers: {total_papers}")

        ## loop over the papers
        for i, paper in papers.iterrows():
            print(f"Processing paper {i+1} of {total_papers}")
            paper_id = paper['id']

            ## check if the paper already exists in the index
            id_match = self.index.fetch([paper_id])
            if len(id_match['vectors']) > 0:
                print('Paper already in index, skipping.\n')
                continue

            ## create embedding of the paper summary
            attempt = 0
            while attempt < max_retries:
                try:
                    res = self.openAIClient.embeddings.create(
                        input=paper['summary'],
                        model=self.embed_model
                    )
                    embedding = res.data[0].embedding
                    break
                except Exception as e:
                    print(f"OpenAI call failure, error {e}.")
                    attempt += 1
                    if attempt < max_retries:
                        print("Trying again.")
                        time.sleep(wait_period)
                    else:
                        print("Max retries reached, failing.")
                        embedding = None
            
            ## if the embedding was sucessful, set the metadata and add to the upsert array as a tuple
            if embedding:
                metadata = {
                    'id': paper_id,
                    'title': paper['title'],
                    'summary': paper['summary'],
                    'published': str(paper['published']),
                    'updated': str(paper['updated'])
                }

                upsert_array.append((paper_id, embedding, metadata))

            ## if the upsert array contains tuples(papers) then upsert them
            if upsert_array:
                print(f"Upserting {len(upsert_array)} papers")
                attempt = 0
                while attempt < max_retries:
                    try:
                        self.index.upsert(vectors=upsert_array, namespace='abstracts')
                        upsert_array = []
                        break
                    except Exception as e:
                        print(f"Upsert failure {e}, waiting.")
                        attempt += 1
                        if attempt < max_retries:
                            print("Trying again.")
                            time.sleep(wait_period)
                        else:
                            print("Max retries reached, failing.")