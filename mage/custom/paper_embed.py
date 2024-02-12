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
    ## create embedding object
    embd = emb(host='host.docker.internal')

    ## create/reset index so it only contains most recent papers
    embd.createIndex()

    ## gets papers from the postgres db with recent papers
    papers = embd.paperQuery()

    ## embeds papers and puts them into pinecone
    embd.embed(papers)
