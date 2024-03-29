import psycopg2
import os
from dotenv import load_dotenv
import pandas as pd

class pg_query:
    def __init__(self, user='postgres', host='localhost'):
        ## get env variables
        load_dotenv()
        
        ## set up postgres params
        self.user = user
        self.host = host
        self.password = os.getenv('postgresPass')
        self.database = 'postgres'

        ## set up postgres connection
        self.conn = psycopg2.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password
        )

    def query(self, query):
        # Create a cursor object
        cur = self.conn.cursor()

        try:
            ## execute the query specified
            cur.execute(query)
            
            ## fetch the results
            rows = cur.fetchall()

        ## 
        except Exception as e:
            print(f"An error occurred: {e}")
        
        ## return results as a data frame
        return pd.DataFrame(rows)