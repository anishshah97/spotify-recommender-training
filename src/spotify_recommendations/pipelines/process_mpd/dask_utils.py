import dask.dataframe as dd
import dask.distributed as d_dist
from dask.distributed import Client


def dask_dedup(ddf, subset, kwargs={}):
    client = d_dist.client._get_global_client() or Client()
    deduped_ddf = ddf.drop_duplicates(
        subset=subset)

    return deduped_ddf
