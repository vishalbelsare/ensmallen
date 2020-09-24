from tqdm.auto import tqdm
from .utils import load_hpo, load_pathway


def test_skipgrams():
    """Test execution of skipgrams."""
    for graph in tqdm((load_hpo(), load_pathway()), desc="Testing Skipgrams", leave=False):
        words, contexts = graph.node2vec(
            batch_size=32,
            length=50,
            window_size=4,
            seed=42,
        )
        assert len(words) == len(contexts)
        edges, labels = graph.link_prediction(
            idx=0,
            batch_size=32
        )
        assert len(edges) == len(labels)
        assert set(labels) <= set([0, 1])