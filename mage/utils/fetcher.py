import requests
import xml.etree.ElementTree as ET
import json
from sqlalchemy import create_engine, text

## this is a simple api call for most recent nlp papers from arxiv, in json format (ID, Title, Summary)
class paperFetcher:
        
        def __init__(self, category='cs.CL', max_results=1, start=0):
                self.base_url = 'http://export.arxiv.org/api/query'
                self.category = category
                self.max_results = max_results
                self.start = start

        ## function to fetch papers from the arxiv api
        def fetchPapers(self):
                query_params = {
                        'search_query': f'cat:{self.category}',
                        'start': self.start,
                        'max_results': self.max_results,
                        'sortBy': 'submittedDate',
                        'sortOrder': 'descending'
                }
                response = requests.get(url=self.base_url, params=query_params)
                return response.text

        ## function to parse papers from the arxiv api
        def parsePapers(self, xml_data):
                root = ET.fromstring(xml_data)
                articles = []

                for paper in root.findall('{http://www.w3.org/2005/Atom}entry'):
                        article_id = paper.find('{http://www.w3.org/2005/Atom}id').text
                        title = paper.find('{http://www.w3.org/2005/Atom}title').text
                        summary = paper.find('{http://www.w3.org/2005/Atom}summary').text
                        published = paper.find('{http://www.w3.org/2005/Atom}published').text
                        updated = paper.find('{http://www.w3.org/2005/Atom}updated').text
                        
                        articles.append(
                                {
                                        'id': article_id, 
                                        'title': title, 
                                        'summary': summary, 
                                        'published': published, 
                                        'updated': updated}
                                )

                return articles

        ## function to get papers in json format
        def getPapersJson(self):
                xml_data = self.fetchPapers()
                articles = self.parsePapers(xml_data)
                return json.dumps(articles, indent=4)
        
        ## function to write papers to the db
        def writeToDb(self, dbname='postgres', user='postgres', password='', host='localhost', df=None, table='raw_papers'):
                engine = create_engine('postgresql://{user}:{password}@{host}:5432/{database}'.format(user=user, password=password, host=host, database=dbname))
                max_updated = None
                with engine.connect() as conn:
                        query = text('select max(updated) from {table}'.format(table=table))
                        result = conn.execute(query)
                        max_updated = result.scalar()
                
                if max_updated == None:
                        df.to_sql(table, engine, if_exists='append', index=False)
                else:
                        df = df[df['updated'] > max_updated]
                        df.to_sql(table, engine, if_exists='append', index=False)

        ## function for removing old papers from the db
        def removeOld(self, dbname='postgres', user='postgres', password='', host='localhost', table='raw_papers', timeframe=7):
                engine = create_engine('postgresql://{user}:{password}@{host}:5432/{database}'.format(user=user, password=password, host=host, database=dbname))

                query = text("delete from {table} where updated::timestamp < current_date - {timeframe}".format(table=table, timeframe=timeframe))
                with engine.begin() as conn:
                        conn.execute(query)