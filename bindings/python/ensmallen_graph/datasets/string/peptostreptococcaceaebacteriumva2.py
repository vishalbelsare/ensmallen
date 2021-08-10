"""
This file offers the methods to automatically retrieve the graph Peptostreptococcaceae bacterium VA2.

The graph is automatically retrieved from the STRING repository. 


References
---------------------
Please cite the following if you use the data:

```latex
@article{szklarczyk2019string,
    title={STRING v11: protein--protein association networks with increased coverage, supporting functional discovery in genome-wide experimental datasets},
    author={Szklarczyk, Damian and Gable, Annika L and Lyon, David and Junge, Alexander and Wyder, Stefan and Huerta-Cepas, Jaime and Simonovic, Milan and Doncheva, Nadezhda T and Morris, John H and Bork, Peer and others},
    journal={Nucleic acids research},
    volume={47},
    number={D1},
    pages={D607--D613},
    year={2019},
    publisher={Oxford University Press}
}
```
"""
from typing import Dict

from ..automatic_graph_retrieval import AutomaticallyRetrievedGraph
from ...ensmallen_graph import EnsmallenGraph  # pylint: disable=import-error


def PeptostreptococcaceaeBacteriumVa2(
    directed: bool = False,
    preprocess: bool = True,
    verbose: int = 2,
    cache: bool = True,
    cache_path: str = "graphs/string",
    version: str = "links.v11.5",
    **additional_graph_kwargs: Dict
) -> EnsmallenGraph:
    """Return new instance of the Peptostreptococcaceae bacterium VA2 graph.

    The graph is automatically retrieved from the STRING repository.	

    Parameters
    -------------------
    directed: bool = False,
        Wether to load the graph as directed or undirected.
        By default false.
    preprocess: bool = True,
        Whether to preprocess the graph to be loaded in 
        optimal time and memory.
    verbose: int = 2,
        Wether to show loading bars during the retrieval and building
        of the graph.
    cache: bool = True,
        Whether to use cache, i.e. download files only once
        and preprocess them only once.
    cache_path: str = "graphs",
        Where to store the downloaded graphs.
    version: str = "links.v11.5",
        The version of the graph to retrieve.		
	The available versions are:
			- homology.v11.0
			- homology.v11.5
			- physical.links.v11.0
			- physical.links.v11.5
			- links.v11.0
			- links.v11.5
    additional_graph_kwargs: Dict,
        Additional graph kwargs.

    Returns
    -----------------------
    Instace of Peptostreptococcaceae bacterium VA2 graph.

	References
	---------------------
	Please cite the following if you use the data:
	
	```latex
	@article{szklarczyk2019string,
	    title={STRING v11: protein--protein association networks with increased coverage, supporting functional discovery in genome-wide experimental datasets},
	    author={Szklarczyk, Damian and Gable, Annika L and Lyon, David and Junge, Alexander and Wyder, Stefan and Huerta-Cepas, Jaime and Simonovic, Milan and Doncheva, Nadezhda T and Morris, John H and Bork, Peer and others},
	    journal={Nucleic acids research},
	    volume={47},
	    number={D1},
	    pages={D607--D613},
	    year={2019},
	    publisher={Oxford University Press}
	}
	```
    """
    return AutomaticallyRetrievedGraph(
        graph_name="PeptostreptococcaceaeBacteriumVa2",
        repository="string",
        version=version,
        directed=directed,
        preprocess=preprocess,
        verbose=verbose,
        cache=cache,
        cache_path=cache_path,
        additional_graph_kwargs=additional_graph_kwargs
    )()
