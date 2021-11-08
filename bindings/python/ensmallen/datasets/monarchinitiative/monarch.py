"""
This file offers the methods to automatically retrieve the graph Monarch.

The graph is automatically retrieved from the MonarchInitiative repository. 


References
---------------------
Please cite the following if you use the data:

```bib
@article{mungall2017monarch,
  title={The Monarch Initiative: an integrative data and analytic platform connecting phenotypes to genotypes across species},
  author={Mungall, Christopher J and McMurry, Julie A and K{\"o}hler, Sebastian and Balhoff, James P and Borromeo, Charles and Brush, Matthew and Carbon, Seth and Conlin, Tom and Dunn, Nathan and Engelstad, Mark and others},
  journal={Nucleic acids research},
  volume={45},
  number={D1},
  pages={D712--D722},
  year={2017},
  publisher={Oxford University Press}
}
```
"""
from typing import Dict, Optional

from ..automatic_graph_retrieval import AutomaticallyRetrievedGraph
from ...ensmallen import Graph  # pylint: disable=import-error


def Monarch(
    directed: bool = False,
    preprocess: bool = True,
    load_nodes: bool = True,
    automatically_enable_speedups_for_small_graphs: bool = True,
    sort_temporary_directory: Optional[str] = None,
    verbose: int = 2,
    cache: bool = True,
    cache_path: Optional[str] = None,
    cache_path_system_variable: str = "GRAPH_CACHE_DIR",
    version: str = "202103",
    **additional_graph_kwargs: Dict
) -> Graph:
    """Return new instance of the Monarch graph.

    The graph is automatically retrieved from the MonarchInitiative repository.	

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
    version: str = "202103"
        The version of the graph to retrieve.		
	The available versions are:
			- 202012
			- 202103
    additional_graph_kwargs: Dict
        Additional graph kwargs.

    Returns
    -----------------------
    Instace of Monarch graph.

	References
	---------------------
	Please cite the following if you use the data:
	
	```bib
	@article{mungall2017monarch,
	  title={The Monarch Initiative: an integrative data and analytic platform connecting phenotypes to genotypes across species},
	  author={Mungall, Christopher J and McMurry, Julie A and K{\"o}hler, Sebastian and Balhoff, James P and Borromeo, Charles and Brush, Matthew and Carbon, Seth and Conlin, Tom and Dunn, Nathan and Engelstad, Mark and others},
	  journal={Nucleic acids research},
	  volume={45},
	  number={D1},
	  pages={D712--D722},
	  year={2017},
	  publisher={Oxford University Press}
	}
	```
    """
    return AutomaticallyRetrievedGraph(
        graph_name="Monarch",
        repository="monarchinitiative",
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
