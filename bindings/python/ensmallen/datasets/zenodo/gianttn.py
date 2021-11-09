"""
This file offers the methods to automatically retrieve the graph GIANT-TN.

The graph is automatically retrieved from the Zenodo repository. 


References
---------------------
Please cite the following if you use the data:

```bib
@article{liu2020supervised,
  title={Supervised learning is an accurate method for network-based gene classification},
  author={Liu, Renming and Mancuso, Christopher A and Yannakopoulos, Anna and Johnson, Kayla A and Krishnan, Arjun},
  journal={Bioinformatics},
  volume={36},
  number={11},
  pages={3457--3465},
  year={2020},
  publisher={Oxford University Press}
}
```
"""
from typing import Dict, Optional

from ..automatic_graph_retrieval import AutomaticallyRetrievedGraph
from ...ensmallen import Graph  # pylint: disable=import-error


def GiantTN(
    directed: bool = False,
    preprocess: bool = True,
    load_nodes: bool = True,
    automatically_enable_speedups_for_small_graphs: bool = True,
    sort_temporary_directory: Optional[str] = None,
    verbose: int = 2,
    cache: bool = True,
    cache_path: Optional[str] = None,
    cache_path_system_variable: str = "GRAPH_CACHE_DIR",
    version: str = "latest",
    **additional_graph_kwargs: Dict
) -> Graph:
    """Return new instance of the GIANT-TN graph.

    The graph is automatically retrieved from the Zenodo repository.	

    Parameters
    -------------------
    directed: bool = False
        Wether to load the graph as directed or undirected.
        By default false.
    preprocess: bool = True
        Whether to preprocess the graph to be loaded in 
        optimal time and memory.
    load_nodes: bool = True
        Whether to load the nodes vocabulary or treat the nodes
        simply as a numeric range.
    automatically_enable_speedups_for_small_graphs: bool = True
        Whether to enable the Ensmallen time-memory tradeoffs in small graphs
        automatically. By default True, that is, if a graph has less than
        50 million edges. In such use cases the memory expenditure is minimal.
    sort_temporary_directory: Optional[str] = None
        Which folder to use to store the temporary files needed to sort in 
        parallel the edge list when building the optimal preprocessed file.
        This defaults to the same folder of the edge list when no value is 
        provided.
    verbose: int = 2
        Wether to show loading bars during the retrieval and building
        of the graph.
    cache: bool = True
        Whether to use cache, i.e. download files only once
        and preprocess them only once.
    cache_path: Optional[str] = None
        Where to store the downloaded graphs.
        If no path is provided, first we check the system variable
        provided below is set, otherwise we use the directory `graphs`.
    cache_path_system_variable: str = "GRAPH_CACHE_DIR"
        The system variable with the default graph cache directory.
    version: str = "latest"
        The version of the graph to retrieve.	
    additional_graph_kwargs: Dict
        Additional graph kwargs.

    Returns
    -----------------------
    Instace of GIANT-TN graph.

	References
	---------------------
	Please cite the following if you use the data:
	
	```bib
	@article{liu2020supervised,
	  title={Supervised learning is an accurate method for network-based gene classification},
	  author={Liu, Renming and Mancuso, Christopher A and Yannakopoulos, Anna and Johnson, Kayla A and Krishnan, Arjun},
	  journal={Bioinformatics},
	  volume={36},
	  number={11},
	  pages={3457--3465},
	  year={2020},
	  publisher={Oxford University Press}
	}
	```
    """
    return AutomaticallyRetrievedGraph(
        graph_name="GiantTN",
        repository="zenodo",
        version=version,
        directed=directed,
        preprocess=preprocess,
        load_nodes=load_nodes,
        verbose=verbose,
        cache=cache,
        cache_path=cache_path,
        cache_path_system_variable=cache_path_system_variable,
        additional_graph_kwargs=additional_graph_kwargs
    )()
