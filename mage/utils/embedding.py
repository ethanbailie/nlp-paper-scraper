import psycopg2
import os
import time
from openai import OpenAI
import pinecone
from dotenv import load_dotenv
import pandas as pd

class embedder:
    def __init__(self, user='postgres', host='localhost', openai_key=None, pinecone_key=None):
        # get env variables
        load_dotenv()
        
        # set up postgres connector
        self.user = user
        self.host = host
        self.password = os.getenv('postgresPass')
        self.database = 'postgres'

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
        self.env = "gcp-starter"
        self.index_name = 'nlp-embedding'

        # initialize openai client
        self.openAIClient = OpenAI(
            api_key=self.openai_key
        )

        # initialize pinecone
        pinecone.init(api_key=self.pinecone_key, environment=self.env)
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
        # grabs all papers in db
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
        # instantiate upsert
        upsert_array = []

        # set starting point
        index_stats = self.index.describe_index_stats()
        starting_point = index_stats['total_vector_count'] - 3

        # retries for if the call fails
        max_retries = 3
        wait_period = 5

        # embeds papers
        for i in range(starting_point,len(papers)):
            
            # grab paper
            paper = papers.iloc[i]
            
            # create id
            paper_id = paper['id']
            
            # check if the paper already exists in the index
            # if it does, skip remainder
            id_match = self.index.fetch([paper_id])
            if len(id_match['vectors']) > 0:
                print('Review already in index, skipping.')
                continue

            # create embedding
            attempt = 0
            while attempt < max_retries:
                
                try:
                    res = self.openAIClient.embeddings.create(
                        input=[paper['title'], paper['summary']],
                        model=self.embed_model
                    )
                    embedding = res.data[0].embedding
                    break

                except Exception as e:
                    print("OpenAI call failure, waiting.")
                    attempt += 1
                    if attempt < max_retries:
                        print("Trying again.")
                        time.sleep(wait_period)
                    else:
                        print("Max retries reached, failing.")
        
            metadata = {
                'id': paper['id'],
                'title': paper['title'],
                'summary': paper['summary'],
                'published': str(paper['published']),
                'updated': str(paper['updated'])
            }

            upsert_array.append((embedding, metadata))

            if i >= 0 and (i % 50) == 0:
                print("Index Status:")
                print(self.index.describe_index_stats())
                print("Papers Processed:")
                print(i)

                attempt = 0
                while attempt < max_retries:
                    try:
                        self.index.upsert(upsert_array)
                        upsert_array = []
                        time.sleep(0.1)
                        break
                        
                    except Exception as e:
                        print("Upsert failure, waiting.")
                        attempt += 1
                        if attempt < max_retries:
                            print("Trying again.")
                            time.sleep(wait_period)
                        else:
                            print("Max retries reached, failing.")
