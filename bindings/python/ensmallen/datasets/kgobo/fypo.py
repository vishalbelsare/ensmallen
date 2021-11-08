"""
This file offers the methods to automatically retrieve the graph FYPO.

The graph is automatically retrieved from the KGOBO repository. 


References
---------------------
Please cite the following if you use the data:

```bib
@misc{kgobo,
  title        = "KG-OBO",
  year         = "2021",
  author       = "{Reese, Justin and Caufield, Harry}",
  howpublished = {\\url{https://github.com/Knowledge-Graph-Hub/kg-obo}},
  note = {Online; accessed 14 September 2021}
}
```
"""
from typing import Dict

from ..automatic_graph_retrieval import AutomaticallyRetrievedGraph
from ...ensmallen import Graph  # pylint: disable=import-error


def FYPO(
    directed: bool = False,
    preprocess: bool = True,
    load_nodes: bool = True,
    verbose: int = 2,
    cache: bool = True,
    cache_path: Optional[str] = None,
    cache_path_system_variable: str = "GRAPH_CACHE_DIR",
    version: str = "2021-10-05",
    **additional_graph_kwargs: Dict
) -> Graph:
    """Return new instance of the FYPO graph.

    The graph is automatically retrieved from the KGOBO repository.	

    Parameters
    -------------------
    directed: bool = False
        Wether to load the graph as directed or undirected.
        By default false.
    preprocess: bool = True
        Whether to preprocess the graph to be loaded in 
        optimal time and memory.
    load_nodes: bool = True,
        Whether to load the nodes vocabulary or treat the nodes
        simply as a numeric range.
    verbose: int = 2,
        Wether to show loading bars during the retrieval and building
        of the graph.
    cache: bool = True
        Whether to use cache, i.e. download files only once
        and preprocess them only once.
    cache_path: Optional[str] = None,
        Where to store the downloaded graphs.
        If no path is provided, first we check the system variable
        provided below is set, otherwise we use the directory `graphs`.
    cache_path_system_variable: str = "GRAPH_CACHE_DIR",
        The system variable with the default graph cache directory.
    version: str = "2021-10-05"
        The version of the graph to retrieve.		
	The available versions are:
			- 2021-10-05
    additional_graph_kwargs: Dict
        Additional graph kwargs.

    Returns
    -----------------------
    Instace of FYPO graph.

	References
	---------------------
	Please cite the following if you use the data:
	
	```bib
	@misc{kgobo,
	  title        = "KG-OBO",
	  year         = "2021",
	  author       = "{Reese, Justin and Caufield, Harry}",
	  howpublished = {\\url{https://github.com/Knowledge-Graph-Hub/kg-obo}},
	  note = {Online; accessed 14 September 2021}
	}
	```
    """
    return AutomaticallyRetrievedGraph(
        graph_name="FYPO",
        repository="kgobo",
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
