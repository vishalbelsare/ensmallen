use numpy::{PyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyAttributeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{HashMap, HashSet};

#[allow(unused_imports)]
use graph::{EdgeT, EdgeTypeT, NodeT, NodeTypeT, Result, WeightT};
use tags::*;

mod macros;
pub(crate) use crate::macros::*;
mod express_measures;
pub(crate) use crate::express_measures::*;

mod node2vec;
pub use crate::node2vec::*;

mod edge_prediction_perceptron;
pub(crate) use edge_prediction_perceptron::*;

mod basic_embedding_model_binding;
pub(crate) use basic_embedding_model_binding::*;

mod basic_siamese_model_binding;
pub(crate) use basic_siamese_model_binding::*;

mod dense;
pub use dense::*;

mod graph_embedder;
pub use graph_embedder::*;

mod edge_file_writer;
mod hash;
mod node_file_writer;
mod preprocessing;
mod subgraphs;
mod trees;
mod utilities;
pub(crate) use crate::preprocessing::*;
pub(crate) use crate::utilities::*;
mod types;
pub(crate) use crate::types::*;
mod operators;
mod spine;
pub(crate) use spine::*;
mod weighted_spine;
pub(crate) use weighted_spine::*;
mod walks;

#[pymodule]
fn models(_py: Python, _m: &PyModule) -> PyResult<()> {
    _m.add_class::<CBOW>()?;
    _m.add_class::<GloVe>()?;
    _m.add_class::<SkipGram>()?;
    _m.add_class::<WalkletsCBOW>()?;
    _m.add_class::<WalkletsGloVe>()?;
    _m.add_class::<WalkletsSkipGram>()?;
    _m.add_class::<TransE>()?;
    _m.add_class::<TransH>()?;
    _m.add_class::<Unstructured>()?;
    _m.add_class::<FirstOrderLINE>()?;
    _m.add_class::<SecondOrderLINE>()?;
    _m.add_class::<SPINE>()?;
    _m.add_class::<WeightedSPINE>()?;
    _m.add_class::<EdgePredictionPerceptron>()?;
    Ok(())
}

// automatically generated files
mod auto_generated_bindings;
pub use auto_generated_bindings::*;
