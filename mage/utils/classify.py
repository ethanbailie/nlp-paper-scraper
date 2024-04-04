import psycopg2
import os
from openai import OpenAI
import pinecone
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text

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

        self.conn = psycopg2.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password
        )

        self.cur = self.conn.cursor()

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

    ## performs semantic search
    def semanticSearch(self, namespace, searchInput, topKInput=100):
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
                    namespace=namespace,
                    vector=searchEmbedding, 
                    top_k=topKInput,
                    include_metadata = True
                )

                matches = searchResult['matches']
                return matches
            
            except Exception as e:
                print(e)
                print("Search failed.")
                count +=1

        
    
    ## this function performs semantic search to find papers associated with a category
    def classifier(self):
        ## create temp table for raw scores
        self.cur.execute("""
            create table if not exists public.temp_scores (
                id varchar primary key,
                foundational_models_and_architectures numeric,
                language_understanding_and_generation numeric,
                transfer_learning_and_adaptation numeric,
                multilingual_and_cross_lingual_models numeric,
                ethics_bias_and_fairness numeric,
                interpretability_and_explainability numeric,
                evaluation_and_benchmarks numeric,
                information_retrieval_and_extraction numeric,
                applications_of_nlp_and_llms numeric,
                resource_efficient_models_and_deployment numeric,
                tokenization_and_text_processing numeric
            )
        """)
        self.conn.commit()
        
        ## categories for the postgres table
        postgres_categories = [
            'foundational_models_and_architectures',
            'language_understanding_and_generation',
            'transfer_learning_and_adaptation',
            'multilingual_and_cross_lingual_models',
            'ethics_bias_and_fairness',
            'interpretability_and_explainability',
            'evaluation_and_benchmarks',
            'information_retrieval_and_extraction',
            'applications_of_nlp_and_llms',
            'resource_efficient_models_and_deployment',
            'tokenization_and_text_processing'
        ]

        ## descriptions of each category that will be used to search for relevant papers
        descriptions = [
            # Foundational Models and Architectures
            "Exploration of core AI model innovations, specifically in BERT, GPT, and transformer technologies, focusing on architectural improvements, efficiency optimization, and the principles governing model scaling.",
            
            # Language Understanding and Generation
            "Investigations into enhancing AI's capability for understanding complex language structures and generating coherent, contextually aware text, including creative outputs such as poetry and narrative content.",
            
            # Transfer Learning and Adaptation
            "Discussion on methodologies for knowledge adaptation and application across different domains, emphasizing fine-tuning for specialized tasks and strategies for effective few-shot and zero-shot learning approaches.",
            
            # Multilingual and Cross-Lingual Models
            "Development and assessment of AI models proficient in multiple languages, focusing on cross-lingual knowledge transfer techniques and the training of multilingual systems.",
            
            # Ethics, Bias, and Fairness
            "Examination of the societal implications of NLP and large language models, with a focus on identifying, mitigating biases, and promoting ethical practices, fairness, and transparency in AI development.",
            
            # Interpretability and Explainability
            "Research into methodologies for enhancing the clarity and comprehensibility of AI model decisions, including innovative approaches to visualize and elucidate model processes and behaviors.",
            
            # Evaluation and Benchmarks
            "Creation and refinement of benchmarks, metrics, and datasets for the rigorous evaluation of NLP model performance across various tasks and real-world applications.",
            
            # Information Retrieval and Extraction
            "Techniques and algorithms for the extraction of structured data from unstructured text sources, enhancing capabilities in search, summarization, question answering, and topic discovery.",
            
            # Applications of NLP and LLMs
            "Case studies and research on applying NLP and large language models within specific sectors, including healthcare, finance, legal, and educational, addressing domain-specific challenges and opportunities.",
            
            # Resource-Efficient Models and Deployment
            "Examination of resource-efficient models and deployments for reducing the resource demands of AI models, covering strategies for model compression, quantization, and deployment in environments with limited computational resources.",
            
            # Tokenization and Text Processing
            "Research on tokenization algorithms and encoder model enhancements focused on optimizing text semantic representation, efficient preprocessing techniques, and overall improvements in NLP task performance through advanced text processing pipelines."]

        ## counter to track current category
        counter = 0

        ## perform semantic search for each category
        for description in descriptions:
            results = self.semanticSearch(namespace='abstracts', searchInput=description)

            ## for every match, assign the semantic search score to the ID
            ids = [match['id'] for match in results]
            scores = [match['score'] for match in results]

            ## creates a dataframe with the IDs and scores for a given category
            category_df = pd.DataFrame({'id': ids, 'scores': scores})

            for index, row in category_df.iterrows():
                self.cur.execute("""
                    insert into public.temp_scores (id, {0})
                    values (%s, %s)
                    on conflict (id) do update set {0} = excluded.{0}
                """.format(postgres_categories[counter]), (row['id'], row['scores']))
                self.conn.commit()
            
            counter += 1

        ## turn nulls to 0s so the vectors later are valid
        self.cur.execute("""
            update public.temp_scores
            set foundational_models_and_architectures = coalesce(foundational_models_and_architectures, 0),
                language_understanding_and_generation = coalesce(language_understanding_and_generation, 0),
                transfer_learning_and_adaptation = coalesce(transfer_learning_and_adaptation, 0),
                multilingual_and_cross_lingual_models = coalesce(multilingual_and_cross_lingual_models, 0),
                ethics_bias_and_fairness = coalesce(ethics_bias_and_fairness, 0),
                interpretability_and_explainability = coalesce(interpretability_and_explainability, 0),
                evaluation_and_benchmarks = coalesce(evaluation_and_benchmarks, 0),
                information_retrieval_and_extraction = coalesce(information_retrieval_and_extraction, 0),
                applications_of_nlp_and_llms = coalesce(applications_of_nlp_and_llms, 0),
                resource_efficient_models_and_deployment = coalesce(resource_efficient_models_and_deployment, 0),
                tokenization_and_text_processing = coalesce(tokenization_and_text_processing, 0)
        """)
        self.conn.commit()

        ## get the raw scores as a dataframe
        self.cur.execute('select * from public.temp_scores')
        raw_scores = pd.DataFrame(self.cur.fetchall())
        col_names = [desc[0] for desc in self.cur.description]
        raw_scores.columns = col_names

        ## melt the dataframe to get the scores in a single vector
        stg_only_scores = raw_scores.melt(id_vars='id', var_name='category', value_name='scores')
        only_scores = stg_only_scores.groupby('id')['scores'].apply(list).reset_index(name='scores')

        ## get the papers as a dataframe
        self.cur.execute('select id, title, updated from public.raw_papers')
        papers = pd.DataFrame(self.cur.fetchall())
        col_names = [desc[0] for desc in self.cur.description]
        papers.columns = col_names

        ## join the scores and papers
        scores = pd.merge(only_scores, papers, on='id', how='inner')

        ## drop the temp table
        self.cur.execute('drop table public.temp_scores')
        self.conn.commit()

        return scores

        
    ## this function writes the scores to the database
    def writeScores(self, scores):
        engine = create_engine(f'postgresql://{self.user}:{self.password}@{self.host}:5432/{self.database}')
        scores.to_sql('scores', engine, if_exists='replace', index=False)