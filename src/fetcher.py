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

        def fetch_papers(self):
                query_params = {
                        'search_query': f'cat:{self.category}',
                        'start': self.start,
                        'max_results': self.max_results,
                        'sortBy': 'submittedDate',
                        'sortOrder': 'descending'
                }
                response = requests.get(url=self.base_url, params=query_params)
                return response.text

        def parse_papers(self, xml_data):
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

        def get_papers_json(self):
                xml_data = self.fetch_papers()
                articles = self.parse_papers(xml_data)
                return json.dumps(articles, indent=4)
        
        def write_to_db(self, dbname='postgres', user='postgres', password='', host='localhost', df=None, table='raw_papers'):
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
                