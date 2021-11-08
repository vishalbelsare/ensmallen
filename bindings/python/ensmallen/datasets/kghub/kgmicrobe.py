"""
This file offers the methods to automatically retrieve the graph kg-microbe.

The graph is automatically retrieved from the KGHub repository. 


References
---------------------
Please cite the following if you use the data:

```bib
@article{joachimiakkg,
  title={KG-Microbe: a reference knowledge-graph and platform for harmonized microbial information},
  author={Joachimiak, Marcin P and Reese, Justin T and Hegde, Harshad and Cappelletti, Luca and Mungall, Christopher J and Duncan, William D and Thessen, Anne E}
}
```
"""
from typing import Dict, Optional

from ..automatic_graph_retrieval import AutomaticallyRetrievedGraph
from ...ensmallen import Graph  # pylint: disable=import-error


def KGMicrobe(
    directed: bool = False,
    preprocess: bool = True,
    load_nodes: bool = True,
    verbose: int = 2,
    cache: bool = True,
    cache_path: Optional[str] = None,
    cache_path_system_variable: str = "GRAPH_CACHE_DIR",
    version: str = "current",
    **additional_graph_kwargs: Dict
) -> Graph:
    """Return new instance of the kg-microbe graph.

    The graph is automatically retrieved from the KGHub repository.	

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
    version: str = "current"
        The version of the graph to retrieve.		
	The available versions are:
			- 20210422
			- 20210517
			- 20210608
			- 20210615
			- 20210617
			- 20210622
			- 20210715
			- current
    additional_graph_kwargs: Dict
        Additional graph kwargs.

    Returns
    -----------------------
    Instace of kg-microbe graph.

	References
	---------------------
	Please cite the following if you use the data:
	
	```bib
	@article{joachimiakkg,
	  title={KG-Microbe: a reference knowledge-graph and platform for harmonized microbial information},
	  author={Joachimiak, Marcin P and Reese, Justin T and Hegde, Harshad and Cappelletti, Luca and Mungall, Christopher J and Duncan, William D and Thessen, Anne E}
	}
	```
    """
    return AutomaticallyRetrievedGraph(
        graph_name="KGMicrobe",
        repository="kghub",
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
