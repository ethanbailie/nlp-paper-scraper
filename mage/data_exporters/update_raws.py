from mage_ai.settings.repo import get_repo_path
from mage_ai.io.config import ConfigFileLoader
from mage_ai.io.postgres import Postgres
from pandas import DataFrame
from os import path

from mage.utils.fetcher import paperFetcher as pf

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data_to_postgres(df: DataFrame, **kwargs) -> None:
    """
    write papers from previous block to both databases (latest and historical)
    and removes old papers from the database holding latest papers
    """
    pf().write_to_db(host='host.docker.internal', df=df, table='all_papers')
    
    pf().write_to_db(host='host.docker.internal', df=df)
    
    pf().remove_old(host='host.docker.internal', timeframe=7)
