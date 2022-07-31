use super::mmap_numpy_npy::{
    create_memory_mapped_numpy_array, load_memory_mapped_numpy_array, Dtype,
};
use super::*;
use cpu_models::{AnchorFeatureTypes, AnchorTypes, AnchorsInferredNodeEmbeddingModel, BasicSPINE};
use numpy::{PyArray1, PyArray2};
use rayon::prelude::*;
use std::convert::TryFrom;
use types::ThreadDataRaceAware;

#[derive(Debug, Clone)]
pub struct BasicSPINEBinding<Model, const AFT: AnchorFeatureTypes, const AT: AnchorTypes>
where
    Model: AnchorsInferredNodeEmbeddingModel<AT, AFT>,
{
    pub inner: Model,
    pub path: Option<String>,
}

impl FromPyDict for BasicSPINE {
    fn from_pydict(py_kwargs: Option<&PyDict>) -> PyResult<Self> {
        let py = pyo3::Python::acquire_gil();
        let kwargs = normalize_kwargs!(py_kwargs, py.python());

        pe!(validate_kwargs(
            kwargs,
            &["embedding_size", "maximum_depth", "path", "verbose"]
        ))?;

        pe!(BasicSPINE::new(
            extract_value_rust_result!(kwargs, "embedding_size", usize),
            extract_value_rust_result!(kwargs, "maximum_depth", usize),
            extract_value_rust_result!(kwargs, "verbose", bool),
        ))
    }
}

impl<Model, const AFT: AnchorFeatureTypes, const AT: AnchorTypes> FromPyDict
    for BasicSPINEBinding<Model, AFT, AT>
where
    Model: AnchorsInferredNodeEmbeddingModel<AT, AFT>,
    Model: From<BasicSPINE>,
{
    fn from_pydict(py_kwargs: Option<&PyDict>) -> PyResult<Self> {
        Ok(Self {
            inner: BasicSPINE::from_pydict(py_kwargs)?.into(),
            path: match py_kwargs {
                None => None,
                Some(kwargs) => {
                    extract_value_rust_result!(kwargs, "path", String)
                }
            },
        })
    }
}

macro_rules! impl_spine_embedding {
    ($($dtype:ty : $dtype_enum:expr),*) => {
        impl<Model, const AFT: AnchorFeatureTypes, const AT: AnchorTypes> BasicSPINEBinding<Model, AFT, AT> where
            Model: AnchorsInferredNodeEmbeddingModel<AT, AFT>,
        {
            /// Return numpy embedding curresponding to the provided indices.
            ///
            /// Parameters
            /// --------------
            /// node_ids: np.ndarray
            ///     Numpy vector with node IDs to be queried.
            ///
            /// Raises
            /// --------------
            /// ValueError
            ///     If the path was not provided to the constructor.
            /// ValueError
            ///     If no embedding exists at the provided path.
            fn get_mmap_node_embedding_from_node_ids(
                &self,
                node_ids: Py<PyArray1<NodeT>>
            ) -> PyResult<Py<PyAny>> {
                if self.path.is_none() {
                    return pe!(Err(
                        format!(
                            concat!(
                                "The current instance of {} ",
                                "was not instantiated with a mmap path."
                            ),
                            self.inner.get_model_name()
                        )
                    ));
                }

                let gil = pyo3::Python::acquire_gil();
                let node_ids = node_ids.as_ref(gil.python());
                let node_ids_ref = unsafe { node_ids.as_slice()? };

                let (embedding_dtype, embedding) = load_memory_mapped_numpy_array(
                    gil.python(),
                    self.path.as_ref().map(|x| x.as_str())
                );

                match pe!(Dtype::try_from(embedding_dtype))?.to_string().as_str() {
                    $(
                        stringify!($dtype) => {
                            let casted_embedding = embedding.cast_as::<PyArray2<$dtype>>(gil.python())?;
                            let number_of_nodes: usize = casted_embedding.shape()[0] as usize;
                            let embedding_size: usize = casted_embedding.shape()[1] as usize;
                            let embedding_slice = unsafe { casted_embedding.as_slice()? };
                            let result:  &PyArray2<$dtype> = unsafe{PyArray2::new(gil.python(), [node_ids.len(), embedding_size], false)};
                            let shared_result_slice = ThreadDataRaceAware {
                                t: result,
                            };
                            embedding_slice.as_ref().par_chunks(number_of_nodes).enumerate().for_each(|(feature_number, feature)|{
                                node_ids_ref.iter().for_each(|&node_id| unsafe {
                                    *(shared_result_slice.t.uget_mut([node_id as usize, feature_number])) = feature[node_id as usize];
                                });
                            });
                            Ok(result.to_owned().into())
                        }
                    )*
                    dtype => pe!(Err(format!(
                        concat!(
                            "The provided dtype {:?} is not supported. The supported ",
                            "data types are `u8`, `u16`, `u32` and `u64`."
                        ),
                        dtype
                    ))),
                }
            }

            /// Return numpy embedding with SPINE node embedding.
            ///
            /// Do note that the embedding is returned transposed.
            ///
            /// Parameters
            /// --------------
            /// graph: Graph
            ///     The graph to embed.
            /// dtype: Optional[str] = None
            ///     Dtype to use for the embedding. Note that an improper dtype may cause overflows.
            ///     When not provided, we automatically infer the best one by using the diameter.
            fn fit_transform(
                &self,
                graph: &Graph,
                py_kwargs: Option<&PyDict>,
            ) -> PyResult<Py<PyAny>> {
                let gil = pyo3::Python::acquire_gil();
                let kwargs = normalize_kwargs!(py_kwargs, gil.python());

                pe!(validate_kwargs(
                    kwargs,
                    &["dtype",]
                ))?;

                let verbose = extract_value_rust_result!(kwargs, "verbose", bool);
                let dtype = match extract_value_rust_result!(kwargs, "dtype", &str) {
                    Some(dtype) => dtype,
                    None => {
                        let (max_u8, max_u16, max_u32) = (u8::MAX as usize, u16::MAX as usize, u32::MAX as usize);
                        match pe!(graph.get_diameter(Some(true), verbose))? as usize {
                            x if (0..=max_u8).contains(&x) => "u8",
                            x if (max_u8..=max_u16).contains(&x) => "u16",
                            x if (max_u16..=max_u32).contains(&x) => "u32",
                            _ => "u64",
                        }
                    }
                };

                let rows_number = graph.inner.get_number_of_nodes() as isize;
                let columns_number = pe!(self.inner.get_embedding_size(&graph.inner))? as isize;
                match dtype {
                    $(
                        stringify!($dtype) => {
                            let embedding = create_memory_mapped_numpy_array(
                                gil.python(),
                                self.path.as_ref().map(|x| x.as_str()),
                                $dtype_enum,
                                vec![rows_number, columns_number],
                                true,
                            );

                            let s = embedding.cast_as::<PyArray2<$dtype>>(gil.python())?;

                            let embedding_slice = unsafe { s.as_slice_mut()? };

                            pe!(self.inner.fit_transform(
                                &graph.inner,
                                embedding_slice,
                            ))?;

                            Ok(embedding)
                        }
                    )*
                    dtype => pe!(Err(format!(
                        concat!(
                            "The provided dtype {} is not supported. The supported ",
                            "data types are `u8`, `u16`, `u32` and `u64`."
                        ),
                        dtype
                    ))),
                }
            }
        }
    };
}

impl_spine_embedding! {
    u8 : Dtype::U8,
    u16: Dtype::U16,
    u32: Dtype::U32,
    u64: Dtype::U64
}

#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(text_signature = "(*, embedding_size, maximum_depth, verbose)")]
pub struct DegreeSPINE {
    pub inner: BasicSPINEBinding<
        cpu_models::DegreeSPINE,
        { AnchorFeatureTypes::ShortestPaths },
        { AnchorTypes::Degrees },
    >,
}

#[pymethods]
impl DegreeSPINE {
    #[new]
    #[args(py_kwargs = "**")]
    /// Return a new instance of the DegreeSPINE model.
    ///
    /// Parameters
    /// ------------------------
    /// embedding_size: int = 100
    ///     Size of the embedding.
    /// maximum_depth: Optional[int] = None
    ///     Maximum depth of the shortest path.
    /// verbose: bool = True
    ///     Whether to show loading bars.
    pub fn new(py_kwargs: Option<&PyDict>) -> PyResult<DegreeSPINE> {
        Ok(Self {
            inner: BasicSPINEBinding::from_pydict(py_kwargs)?,
        })
    }

    #[pyo3(text_signature = "($self, node_ids)")]
    /// Return numpy embedding curresponding to the provided indices.
    ///
    /// Parameters
    /// --------------
    /// node_ids: np.ndarray
    ///     Numpy vector with node IDs to be queried.
    ///
    /// Raises
    /// --------------
    /// ValueError
    ///     If the path was not provided to the constructor.
    /// ValueError
    ///     If no embedding exists at the provided path.
    fn get_mmap_node_embedding_from_node_ids(
        &self,
        node_ids: Py<PyArray1<NodeT>>,
    ) -> PyResult<Py<PyAny>> {
        self.inner.get_mmap_node_embedding_from_node_ids(node_ids)
    }

    #[args(py_kwargs = "**")]
    #[pyo3(text_signature = "($self, graph, *, dtype)")]
    /// Return numpy embedding with Degree SPINE node embedding.
    ///
    /// Do note that the embedding is returned transposed.
    ///
    /// Parameters
    /// --------------
    /// graph: Graph
    ///     The graph to embed.
    /// dtype: Optional[str] = None
    ///     Dtype to use for the embedding. Note that an improper dtype may cause overflows.
    ///     When not provided, we automatically infer the best one by using the diameter.
    fn fit_transform(&self, graph: &Graph, py_kwargs: Option<&PyDict>) -> PyResult<Py<PyAny>> {
        self.inner.fit_transform(graph, py_kwargs)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(text_signature = "(*, embedding_size, maximum_depth, verbose)")]
pub struct NodeLabelSPINE {
    pub inner: BasicSPINEBinding<
        cpu_models::NodeLabelSPINE,
        { AnchorFeatureTypes::ShortestPaths },
        { AnchorTypes::NodeTypes },
    >,
}

#[pymethods]
impl NodeLabelSPINE {
    #[new]
    #[args(py_kwargs = "**")]
    /// Return a new instance of the NodeLabelSPINE model.
    ///
    /// Parameters
    /// ------------------------
    /// embedding_size: int = 100
    ///     Size of the embedding.
    /// maximum_depth: Optional[int] = None
    ///     Maximum depth of the shortest path.
    /// verbose: bool = True
    ///     Whether to show loading bars.
    pub fn new(py_kwargs: Option<&PyDict>) -> PyResult<NodeLabelSPINE> {
        Ok(Self {
            inner: BasicSPINEBinding::from_pydict(py_kwargs)?,
        })
    }

    #[pyo3(text_signature = "($self, node_ids)")]
    /// Return numpy embedding curresponding to the provided indices.
    ///
    /// Parameters
    /// --------------
    /// node_ids: np.ndarray
    ///     Numpy vector with node IDs to be queried.
    ///
    /// Raises
    /// --------------
    /// ValueError
    ///     If the path was not provided to the constructor.
    /// ValueError
    ///     If no embedding exists at the provided path.
    fn get_mmap_node_embedding_from_node_ids(
        &self,
        node_ids: Py<PyArray1<NodeT>>,
    ) -> PyResult<Py<PyAny>> {
        self.inner.get_mmap_node_embedding_from_node_ids(node_ids)
    }

    #[args(py_kwargs = "**")]
    #[pyo3(text_signature = "($self, graph, *, dtype)")]
    /// Return numpy embedding with Degree SPINE node embedding.
    ///
    /// Do note that the embedding is returned transposed.
    ///
    /// Parameters
    /// --------------
    /// graph: Graph
    ///     The graph to embed.
    /// dtype: Optional[str] = None
    ///     Dtype to use for the embedding. Note that an improper dtype may cause overflows.
    ///     When not provided, we automatically infer the best one by using the diameter.
    fn fit_transform(&self, graph: &Graph, py_kwargs: Option<&PyDict>) -> PyResult<Py<PyAny>> {
        self.inner.fit_transform(graph, py_kwargs)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(text_signature = "(*, embedding_size, maximum_depth, path, verbose)")]
pub struct ScoreSPINE {
    inner: BasicSPINE,
    path: Option<String>,
}

#[pymethods]
impl ScoreSPINE {
    #[new]
    #[args(py_kwargs = "**")]
    /// Return a new instance of the ScoreSPINE model.
    ///
    /// Parameters
    /// ------------------------
    /// embedding_size: int = 100
    ///     Size of the embedding.
    /// maximum_depth: Optional[int] = None
    ///     Maximum depth of the shortest path.
    /// path: Optional[str] = None
    ///     If passed, create a `.npy` file which will be mem-mapped
    ///     to allow processing embeddings that do not fit in RAM
    /// verbose: bool = True
    ///     Whether to show loading bars.
    pub fn new(py_kwargs: Option<&PyDict>) -> PyResult<ScoreSPINE> {
        Ok(Self {
            inner: BasicSPINE::from_pydict(py_kwargs)?,
            path: match py_kwargs {
                None => None,
                Some(kwargs) => {
                    extract_value_rust_result!(kwargs, "path", String)
                }
            },
        })
    }

    #[pyo3(text_signature = "($self, node_ids)")]
    /// Return numpy embedding curresponding to the provided indices.
    ///
    /// Parameters
    /// --------------
    /// node_ids: np.ndarray
    ///     Numpy vector with node IDs to be queried.
    ///
    /// Raises
    /// --------------
    /// ValueError
    ///     If the path was not provided to the constructor.
    /// ValueError
    ///     If no embedding exists at the provided path.
    fn get_mmap_node_embedding_from_node_ids(
        &self,
        node_ids: Py<PyArray1<NodeT>>,
    ) -> PyResult<Py<PyAny>> {
        BasicSPINEBinding {
            inner: cpu_models::DegreeSPINE::from(self.inner.clone()),
            path: self.path.clone(),
        }
        .get_mmap_node_embedding_from_node_ids(node_ids)
    }

    #[args(py_kwargs = "**")]
    #[pyo3(text_signature = "($self, scores, graph, *, dtype)")]
    /// Return numpy embedding with Degree SPINE node embedding.
    ///
    /// Do note that the embedding is returned transposed.
    ///
    /// Parameters
    /// --------------
    /// scores: np.ndarray
    ///     Scores to create the node groups.
    /// graph: Graph
    ///     The graph to embed.
    /// dtype: Optional[str] = None
    ///     Dtype to use for the embedding. Note that an improper dtype may cause overflows.
    ///     When not provided, we automatically infer the best one by using the diameter.
    fn fit_transform(
        &self,
        scores: Py<PyArray1<f32>>,
        graph: &Graph,
        py_kwargs: Option<&PyDict>,
    ) -> PyResult<Py<PyAny>> {
        let gil = pyo3::Python::acquire_gil();
        let scores_ref = scores.as_ref(gil.python());
        BasicSPINEBinding {
            inner: cpu_models::ScoreSPINE::new(self.inner.clone(), unsafe {
                scores_ref.as_slice().unwrap()
            }),
            path: self.path.clone(),
        }
        .fit_transform(graph, py_kwargs)
    }
}
