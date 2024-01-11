import io
import pandas as pd
import requests
from mage.utils.fetcher import paperFetcher as pf

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_api(*args, **kwargs):
    """
    Template for loading data from API
    """
    papers_json = pf(max_results=200).get_papers_json()
    if papers_json == '[]':
        return None

    papers_df = pd.read_json(papers_json)
    return papers_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
