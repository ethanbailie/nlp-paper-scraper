if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from mage.utils.embedding import embedder as emb

@custom
def paper_embed(*args, **kwargs):
    """
    args: 
        None

    Returns:
        None
    
    all this needs is for the postgres database to contain some papers
    which are then stored in pinecone as vector embeddings
    """
    # Specify your custom logic here
    embd = emb(host='host.docker.internal')
    papers = embd.paperQuery()
    embd.embed(papers)
