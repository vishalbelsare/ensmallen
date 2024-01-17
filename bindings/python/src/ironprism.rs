use super::mmap_numpy_npy::to_numpy_array;
use super::Graph as PyGraph;
use super::NodeT;
use crate::pe;
use bias_aware_edge_features::{BiasAwareEdgeFeature, EdgeStatus, FeatureCombination};
use core::ops::Range;
use custom_iters::prelude::*;
use edge_graph_convolution::prelude::*;
use edge_weighting::prelude::*;
use hyper_sketching::prelude::*;
use iron_graph::prelude::*;
use matrix::prelude::*;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::convert::TryFrom;
use vec_matrix::VecMatrix;

impl Graph for PyGraph {
    type Node = NodeT;

    fn directed(&self) -> bool {
        self.is_directed()
    }
}

impl NodeStream for PyGraph {
    type NodeIter<'a> = Range<NodeT> where Self: 'a;

    fn nodes(&self) -> Self::NodeIter<'_> {
        0..self.get_number_of_nodes() as NodeT
    }
}

impl ParNodeStream for PyGraph {
    type ParNodeIter<'a> = Range<NodeT> where Self: 'a;

    fn par_nodes(&self) -> Self::ParNodeIter<'_> {
        0..self.get_number_of_nodes() as NodeT
    }
}

impl ExactNodeStream for PyGraph {
    fn number_of_nodes(&self) -> usize {
        self.get_number_of_nodes() as usize
    }
}

impl EdgeStream for PyGraph {
    type CoordinatesIter<'a> = impl Iterator<Item = (NodeT, NodeT)> where Self: 'a;

    fn coordinates(&self) -> Self::CoordinatesIter<'_> {
        self.inner
            .iter_directed_edge_node_ids()
            .map(|(_, src, dst)| (src, dst))
    }
}

impl ExactEdgeStream for PyGraph {
    fn number_of_edges(&self) -> usize {
        self.get_number_of_edges() as usize
    }
}

impl RandomAccessGraph for PyGraph {
    type Successors<'a> = Copied<&'a [NodeT]> where Self: 'a;

    fn successors(&self, node: NodeT) -> Self::Successors<'_> {
        Copied::new(
            self.inner
                .get_unchecked_neighbours_node_ids_from_src_node_id(node),
        )
    }
}

impl ExactRandomAccessGraph for PyGraph {
    fn number_of_successors(&self, node: NodeT) -> usize {
        unsafe { self.inner.get_unchecked_node_degree_from_node_id(node) as usize }
    }

    fn node_has_selfloop(&self, node: NodeT) -> bool {
        self.inner.has_selfloop_from_node_id(node)
    }
}

#[pyclass(module = "ironprism")]
pub struct EdgeGraphConvolutionPy {
    inner: LazyEdgeGraphConvolution<
        PyGraph,
        AbsoluteEdgeWeighting<EagerSymmetricallyNormalizedLaplacian<f32, NodeT>>,
        VecComulativelyWeightedFrontier<NodeT, f32>,
    >,
}

impl EdgeGraphConvolutionPy {
    fn normalize_edges<'a>(
        &'a self,
        sources: &'a PyArray1<NodeT>,
        destinations: &'a PyArray1<NodeT>,
    ) -> PyResult<(&'a [NodeT], &'a [NodeT])> {
        let sources_ref = unsafe {
            sources
                .as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        };

        let destinations_ref = unsafe {
            destinations
                .as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        };

        if sources_ref.len() != destinations_ref.len() {
            return Err(PyValueError::new_err(format!(
                concat!(
                    "The provided sources and destinations do not have the same length. ",
                    "The provided sources have length {} and the provided destinations have length {}."
                ),
                sources_ref.len(),
                destinations_ref.len(),
            )));
        }

        let number_of_nodes = self.get_number_of_nodes() as NodeT;

        // We check that all the provided sources are in the graph.
        if sources_ref.par_iter().any(|&node| node >= number_of_nodes) {
            return Err(PyValueError::new_err(
                "Some of the provided sources are not in the graph.",
            ));
        }

        // We check that all the provided destinations are in the graph.
        if destinations_ref
            .par_iter()
            .any(|&node| node >= number_of_nodes)
        {
            return Err(PyValueError::new_err(
                "Some of the provided destinations are not in the graph.",
            ));
        }

        Ok((sources_ref, destinations_ref))
    }

    fn from_edge_node_ids(
        &self,
        sources: Py<PyArray1<NodeT>>,
        destinations: Py<PyArray1<NodeT>>,
        feature_combination: FeatureCombination,
        edge_status: EdgeStatus,
    ) -> PyResult<Py<PyAny>> {
        let gil = pyo3::Python::acquire_gil();
        let py = gil.python();
        let (sources, destinations) =
            self.normalize_edges(sources.as_ref(py), destinations.as_ref(py))?;

        let features: VecMatrix<[usize; 2], f32> = self.inner.from_edge_ids_par(
            sources
                .par_iter()
                .copied()
                .zip(destinations.par_iter().copied()),
            feature_combination,
            edge_status,
        );

        let shape = features.shape();
        let flat_features: Vec<f32> = features.into();

        pe!(to_numpy_array(gil.python(), flat_features, &shape, false,))
    }

    fn get_number_of_nodes(&self) -> usize {
        self.inner.number_of_nodes()
    }
}

#[pymethods]
impl EdgeGraphConvolutionPy {
    #[new]
    /// Create a new EdgeGraphConvolutionPy.
    ///
    /// Parameters
    /// ----------
    /// hops: int
    ///     The number of hops to consider.
    /// graph: ironprism.Graph
    ///     The graph to use.
    fn new(hops: usize, graph: PyGraph) -> Self {
        let edge_weighting =
            AbsoluteEdgeWeighting::new(EagerSymmetricallyNormalizedLaplacian::new(&graph));
        let inner = LazyEdgeGraphConvolution::new(hops, graph.clone(), edge_weighting);
        Self { inner }
    }

    #[pyo3(text_signature = "($self, node_features)")]
    /// Fit the model to the provided node features.
    ///
    /// Parameters
    /// ----------
    /// node_features: numpy.ndarray
    ///    The node features to fit the model to. Must be a 2D array of floats.
    ///
    fn fit(&mut self, node_features: Py<PyArray2<f32>>) -> PyResult<()> {
        let gil = pyo3::Python::acquire_gil();
        let py = gil.python();
        let node_features = node_features.as_ref(py);

        let shape = node_features.shape();

        let node_features = unsafe {
            node_features
                .as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        };

        let node_features = VecMatrix::new([shape[0], shape[1]], node_features);

        self.inner.fit(&node_features);

        Ok(())
    }

    #[pyo3(text_signature = "($self, sources, destinations)")]
    /// Return edge features without assuming the edges are positive.
    ///
    /// Parameters
    /// ----------
    /// sources: numpy.ndarray
    ///     The source nodes of the edges. Must be a 1D array of Node IDs, which must be u32s.
    /// destinations: numpy.ndarray
    ///     The destination nodes of the edges. Must be a 1D array of Node IDs, which must be u32s.
    /// feature_combination: str
    ///     The feature combination to use.
    fn positive(
        &self,
        sources: Py<PyArray1<NodeT>>,
        destinations: Py<PyArray1<NodeT>>,
        feature_combination: String,
    ) -> PyResult<Py<PyAny>> {
        self.from_edge_node_ids(
            sources,
            destinations,
            pe!(FeatureCombination::try_from(feature_combination))?,
            EdgeStatus::Positive,
        )
    }

    #[pyo3(text_signature = "($self, sources, destinations, feature_combination)")]
    /// Return edge features without assuming the edges are negative.
    ///
    /// Parameters
    /// ----------
    /// sources: numpy.ndarray
    ///     The source nodes of the edges. Must be a 1D array of Node IDs, which must be u32s.
    /// destinations: numpy.ndarray
    ///     The destination nodes of the edges. Must be a 1D array of Node IDs, which must be u32s.
    /// feature_combination: str
    ///     The feature combination to use.
    ///
    fn negative(
        &self,
        sources: Py<PyArray1<NodeT>>,
        destinations: Py<PyArray1<NodeT>>,
        feature_combination: String,
    ) -> PyResult<Py<PyAny>> {
        self.from_edge_node_ids(
            sources,
            destinations,
            pe!(FeatureCombination::try_from(feature_combination))?,
            EdgeStatus::Negative,
        )
    }

    #[pyo3(text_signature = "($self, sources, destinations, feature_combination)")]
    /// Return edge features without assuming whether they are positive or negative.
    ///
    /// Parameters
    /// ----------
    /// sources: numpy.ndarray
    ///     The source nodes of the edges. Must be a 1D array of Node IDs, which must be u32s.
    /// destinations: numpy.ndarray
    ///     The destination nodes of the edges. Must be a 1D array of Node IDs, which must be u32s.
    /// feature_combination: str
    ///     The feature combination to use.
    ///
    fn unknown(
        &self,
        sources: Py<PyArray1<NodeT>>,
        destinations: Py<PyArray1<NodeT>>,
        feature_combination: String,
    ) -> PyResult<Py<PyAny>> {
        self.from_edge_node_ids(
            sources,
            destinations,
            pe!(FeatureCombination::try_from(feature_combination))?,
            EdgeStatus::Unknown,
        )
    }
}

#[pyclass(module = "ironprism")]
pub struct HyperSketchingPy {
    inner: HyperSketching<PyGraph>,
}

impl HyperSketchingPy {
    fn normalize_edges<'a>(
        &'a self,
        sources: &'a PyArray1<NodeT>,
        destinations: &'a PyArray1<NodeT>,
    ) -> PyResult<(&'a [NodeT], &'a [NodeT])> {
        let sources_ref = unsafe {
            sources
                .as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        };

        let destinations_ref = unsafe {
            destinations
                .as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        };

        if sources_ref.len() != destinations_ref.len() {
            return Err(PyValueError::new_err(format!(
                concat!(
                    "The provided sources and destinations do not have the same length. ",
                    "The provided sources have length {} and the provided destinations have length {}."
                ),
                sources_ref.len(),
                destinations_ref.len(),
            )));
        }

        let number_of_nodes = self.get_number_of_nodes() as NodeT;

        // We check that all the provided sources are in the graph.
        if sources_ref.par_iter().any(|&node| node >= number_of_nodes) {
            return Err(PyValueError::new_err(
                "Some of the provided sources are not in the graph.",
            ));
        }

        // We check that all the provided destinations are in the graph.
        if destinations_ref
            .par_iter()
            .any(|&node| node >= number_of_nodes)
        {
            return Err(PyValueError::new_err(
                "Some of the provided destinations are not in the graph.",
            ));
        }

        Ok((sources_ref, destinations_ref))
    }

    fn from_edge_node_ids(
        &self,
        sources: Py<PyArray1<NodeT>>,
        destinations: Py<PyArray1<NodeT>>,
        feature_combination: FeatureCombination,
        edge_status: EdgeStatus,
    ) -> PyResult<Py<PyAny>> {
        let gil = pyo3::Python::acquire_gil();
        let py = gil.python();
        let (sources, destinations) =
            self.normalize_edges(sources.as_ref(py), destinations.as_ref(py))?;

        let features: VecMatrix<[usize; 2], f32> = self.inner.from_edge_ids_par(
            sources
                .par_iter()
                .copied()
                .zip(destinations.par_iter().copied()),
            feature_combination,
            edge_status,
        );

        let shape = features.shape();
        let flat_features: Vec<f32> = features.into();

        pe!(to_numpy_array(gil.python(), flat_features, &shape, false,))
    }

    fn get_number_of_nodes(&self) -> usize {
        self.inner.number_of_nodes()
    }
}

#[pymethods]
impl HyperSketchingPy {
    #[new]
    /// Create a new HyperSketchingPy.
    ///
    /// Parameters
    /// ----------
    /// hops: int
    ///     The number of hops to consider.
    /// normalize: bool
    ///     Whether to normalize the features.
    /// graph: ironprism.Graph
    ///     The graph to use.
    fn new(hops: usize, normalize: bool, graph: PyGraph) -> Self {
        let inner = HyperSketching::new(hops, normalize, graph.clone());
        Self { inner }
    }

    #[pyo3(text_signature = "($self)")]
    /// Fit the model.
    fn fit(&mut self) {
        self.inner.fit();
    }

    #[pyo3(text_signature = "($self, sources, destinations, feature_combination)")]
    /// Return edge features without assuming the edges are positive.
    ///
    /// Parameters
    /// ----------
    /// sources: numpy.ndarray
    ///     The source nodes of the edges. Must be a 1D array of Node IDs, which must be u32s.
    /// destinations: numpy.ndarray
    ///    The destination nodes of the edges. Must be a 1D array of Node IDs, which must be u32s.
    /// feature_combination: str
    ///    The feature combination to use.
    ///
    fn positive(
        &self,
        sources: Py<PyArray1<NodeT>>,
        destinations: Py<PyArray1<NodeT>>,
        feature_combination: String,
    ) -> PyResult<Py<PyAny>> {
        self.from_edge_node_ids(
            sources,
            destinations,
            pe!(FeatureCombination::try_from(feature_combination))?,
            EdgeStatus::Positive,
        )
    }

    #[pyo3(text_signature = "($self, sources, destinations, feature_combination)")]
    /// Return edge features without assuming the edges are negative.
    ///
    /// Parameters
    /// ----------
    /// sources: numpy.ndarray
    ///     The source nodes of the edges. Must be a 1D array of Node IDs, which must be u32s.
    /// destinations: numpy.ndarray
    ///    The destination nodes of the edges. Must be a 1D array of Node IDs, which must be u32s.
    /// feature_combination: str
    ///    The feature combination to use.
    ///
    fn negative(
        &self,
        sources: Py<PyArray1<NodeT>>,
        destinations: Py<PyArray1<NodeT>>,
        feature_combination: String,
    ) -> PyResult<Py<PyAny>> {
        self.from_edge_node_ids(
            sources,
            destinations,
            pe!(FeatureCombination::try_from(feature_combination))?,
            EdgeStatus::Negative,
        )
    }

    #[pyo3(text_signature = "($self, sources, destinations, feature_combination)")]
    /// Return edge features without assuming whether they are positive or negative.
    ///
    /// Parameters
    /// ----------
    /// sources: numpy.ndarray
    ///     The source nodes of the edges. Must be a 1D array of Node IDs, which must be u32s.
    /// destinations: numpy.ndarray
    ///    The destination nodes of the edges. Must be a 1D array of Node IDs, which must be u32s.
    /// feature_combination: str
    ///    The feature combination to use.
    ///
    fn unknown(
        &self,
        sources: Py<PyArray1<NodeT>>,
        destinations: Py<PyArray1<NodeT>>,
        feature_combination: String,
    ) -> PyResult<Py<PyAny>> {
        self.from_edge_node_ids(
            sources,
            destinations,
            pe!(FeatureCombination::try_from(feature_combination))?,
            EdgeStatus::Unknown,
        )
    }
}

pub fn register_ironprism(_py: Python, _m: &PyModule) -> PyResult<()> {
    _m.add_class::<EdgeGraphConvolutionPy>()?;
    _m.add_class::<HyperSketchingPy>()?;
    Ok(())
}
