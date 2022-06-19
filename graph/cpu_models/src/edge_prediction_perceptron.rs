use crate::{EdgeEmbeddingMethod, NodeFeaturesBasedEdgePrediction};
use express_measures::dot_product_sequential_unchecked;
use graph::{Graph, NodeT};
use indicatif::ProgressIterator;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use vec_rand::{random_f32, splitmix64};

#[derive(Clone, Debug)]
pub struct EdgePredictionPerceptron {
    /// The name of the method to use to compute the edge embedding.
    edge_embedding_method_name: EdgeEmbeddingMethod,
    /// The weights of the model.
    weights: Vec<f32>,
    /// The bias of the model.
    bias: f32,
    /// The number of epochs to train the model for.
    number_of_epochs: usize,
    /// Number of samples in a mini-batch. By default 1024.
    number_of_edges_per_mini_batch: usize,
    /// Whether to train this model by sampling only edges with nodes with different node types.
    sample_only_edges_with_heterogeneous_node_types: bool,
    /// Learning rate to use to train the model.
    learning_rate: f32,
    /// The random state to reproduce the model initialization and training.
    random_state: u64,
}

impl EdgePredictionPerceptron {
    /// Return new instance of Perceptron for edge prediction.
    ///
    /// # Arguments
    /// * `edge_embedding_method_name`: Option<EdgeEmbeddingMethod> - The embedding method to use. By default the cosine similarity is used.
    /// * `number_of_epochs`: Option<usize> - The number of epochs to train the model for. By default, 100.
    /// * `number_of_edges_per_mini_batch`: Option<usize> - The number of samples to include for each mini-batch. By default 1024.
    /// * `sample_only_edges_with_heterogeneous_node_types`: Option<bool> - Whether to sample negative edges only with source and destination nodes that have different node types. By default false.
    /// * `learning_rate`: Option<f32> - Learning rate to use while training the model. By default 0.001.
    /// * `random_state`: Option<u64> - The random state to reproduce the model initialization and training. By default, 42.
    pub fn new(
        edge_embedding_method_name: Option<EdgeEmbeddingMethod>,
        number_of_epochs: Option<usize>,
        number_of_edges_per_mini_batch: Option<usize>,
        sample_only_edges_with_heterogeneous_node_types: Option<bool>,
        learning_rate: Option<f32>,
        random_state: Option<u64>,
    ) -> Result<Self, String> {
        let number_of_epochs = number_of_epochs.unwrap_or(100);
        if number_of_epochs == 0 {
            return Err(concat!(
                "The provided number of epochs is zero. ",
                "The number of epochs should be strictly greater than zero."
            )
            .to_string());
        }
        let number_of_edges_per_mini_batch = number_of_edges_per_mini_batch.unwrap_or(1024);
        if number_of_edges_per_mini_batch == 0 {
            return Err(concat!(
                "The provided number of edges per mini-batch is zero. ",
                "The number of edges per mini-batch should be strictly greater than zero."
            )
            .to_string());
        }
        let learning_rate = learning_rate.unwrap_or(0.001);
        if learning_rate <= 0.0 {
            return Err(concat!(
                "The provided learning rate must be a value strictly greater than zero."
            )
            .to_string());
        }

        let edge_embedding_method_name =
            edge_embedding_method_name.unwrap_or(EdgeEmbeddingMethod::CosineSimilarity);
        Ok(Self {
            edge_embedding_method_name,
            weights: Vec::new(),
            bias: 0.0,
            number_of_epochs,
            number_of_edges_per_mini_batch,
            sample_only_edges_with_heterogeneous_node_types:
                sample_only_edges_with_heterogeneous_node_types.unwrap_or(false),
            learning_rate,
            random_state: random_state.unwrap_or(42),
        })
    }

    /// Returns the weights of the model.
    pub fn get_weights(&self) -> Result<Vec<f32>, String> {
        if self.weights.is_empty() {
            return Err(concat!(
                "This model has not been trained yet. ",
                "You should call the `.fit` method first."
            )
            .to_string());
        }
        Ok(self.weights.clone())
    }

    /// Returns the bias of the model.
    pub fn get_bias(&self) -> Result<f32, String> {
        if self.weights.is_empty() {
            return Err(concat!(
                "This model has not been trained yet. ",
                "You should call the `.fit` method first."
            )
            .to_string());
        }
        Ok(self.bias)
    }

    /// Returns the prediction for the provided nodes, edge embedding method and current model.
    ///
    /// # Arguments
    /// `src`: NodeT - The source node whose features are to be extracted.
    /// `dst`: NodeT - The destination node whose features are to be extracted.
    /// `node_features`: &[f32] - The node features to use.
    /// `dimension`: usize - The dimension of the provided node features.
    /// `edge_embedding_method`: fn - Callback to the edge embedding method to use.
    ///
    /// # Safety
    /// In this method we do not execute any checks such as whether the
    /// node features are compatible with the provided node IDs, and therefore
    /// improper parametrization may lead to panic or undefined behaviour.
    unsafe fn get_unsafe_prediction(
        &self,
        src: NodeT,
        dst: NodeT,
        node_features: &[f32],
        dimension: usize,
        method: fn(&[f32], &[f32]) -> Vec<f32>,
    ) -> (Vec<f32>, f32) {
        let src = src as usize;
        let dst = dst as usize;
        let src_features = &node_features[src * dimension..(src + 1) * dimension];
        let dst_features = &node_features[dst * dimension..(dst + 1) * dimension];
        let edge_embedding = method(src_features, dst_features);
        let dot = dot_product_sequential_unchecked(&edge_embedding, &self.weights) + self.bias;
        (edge_embedding, 1.0 / (1.0 + (-dot).exp()))
    }
}

impl NodeFeaturesBasedEdgePrediction for EdgePredictionPerceptron {
    fn fit(
        &mut self,
        graph: &Graph,
        node_features: &[f32],
        dimension: usize,
        verbose: Option<bool>,
        support: Option<&Graph>,
        graph_to_avoid: Option<&Graph>,
    ) -> Result<(), String> {
        self.validate_features(graph, node_features, dimension)?;

        let mut random_state: u64 = splitmix64(self.random_state);
        let scale_factor: f32 = (dimension as f32).sqrt();
        let verbose: bool = verbose.unwrap_or(true);

        // Initialize the model with weights and bias in the range (-1 / sqrt(k), +1 / sqrt(k))
        let get_random_weight = |seed: usize| {
            (2.0 * random_f32(splitmix64(random_state + seed as u64)) - 1.0) / scale_factor
        };
        let edge_dimension = self
            .edge_embedding_method_name
            .get_dimensionality(dimension);
        self.weights = (0..edge_dimension)
            .map(|i| get_random_weight(i))
            .collect::<Vec<f32>>();
        self.bias = get_random_weight(self.weights.len());

        // Depending whether verbosity was requested by the user
        // we create or not a visible progress bar to show the progress
        // in the training epochs.
        let progress_bar = if verbose {
            let pb = ProgressBar::new(self.number_of_epochs as u64);
            pb.set_style(ProgressStyle::default_bar().template(concat!(
                "Edge Prediction Perceptron Epochs ",
                "{spinner:.green} [{elapsed_precise}] ",
                "[{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})"
            )));
            pb
        } else {
            ProgressBar::hidden()
        };

        let number_of_batches_per_epoch = (graph.get_number_of_directed_edges() as f32
            / self.number_of_edges_per_mini_batch as f32)
            .ceil() as usize;

        let method = self.edge_embedding_method_name.get_method();
        let batch_learning_rate: f32 =
            self.learning_rate / self.number_of_edges_per_mini_batch as f32;

        // We start to loop over the required amount of epochs.
        (0..self.number_of_epochs)
            .progress_with(progress_bar)
            .map(|_| {
                (0..number_of_batches_per_epoch)
                    .map(|_| {
                        random_state = splitmix64(random_state);
                        let (total_weights_gradient, total_bias_gradient) = graph
                            .par_iter_edge_prediction_mini_batch(
                                random_state,
                                self.number_of_edges_per_mini_batch,
                                self.sample_only_edges_with_heterogeneous_node_types,
                                Some(0.5),
                                Some(true),
                                None,
                                Some(true),
                                support,
                                graph_to_avoid,
                            )?
                            .map(|(src, dst, label)| {
                                let (mut edge_embedding, prediction) = unsafe {
                                    self.get_unsafe_prediction(
                                        src,
                                        dst,
                                        node_features,
                                        dimension,
                                        method,
                                    )
                                };

                                let variation = if label { prediction - 1.0 } else { prediction };

                                edge_embedding.iter_mut().for_each(|edge_feature| {
                                    *edge_feature *= variation;
                                });

                                (edge_embedding, variation)
                            })
                            .reduce(
                                || (vec![0.0; edge_dimension], 0.0),
                                |(mut total_weights_gradient, mut total_bias_gradient): (
                                    Vec<f32>,
                                    f32,
                                ),
                                 (partial_weights_gradient, partial_bias_gradient): (
                                    Vec<f32>,
                                    f32,
                                )| {
                                    total_weights_gradient
                                        .iter_mut()
                                        .zip(partial_weights_gradient.into_iter())
                                        .for_each(
                                            |(total_weight_gradient, partial_weight_gradient)| {
                                                *total_weight_gradient += partial_weight_gradient;
                                            },
                                        );
                                    total_bias_gradient += partial_bias_gradient;
                                    (total_weights_gradient, total_bias_gradient)
                                },
                            );
                        self.bias -= total_bias_gradient * batch_learning_rate;
                        self.weights
                            .par_iter_mut()
                            .zip(total_weights_gradient.into_par_iter())
                            .for_each(|(weight, total_weight_gradient)| {
                                *weight -= total_weight_gradient * batch_learning_rate;
                            });
                        Ok(())
                    })
                    .collect::<Result<(), String>>()
            })
            .collect::<Result<(), String>>()
    }

    fn is_trained(&self) -> bool {
        !self.weights.is_empty()
    }

    fn get_edge_embedding_method(&self) -> &EdgeEmbeddingMethod {
        &self.edge_embedding_method_name
    }

    fn get_input_size(&self) -> usize {
        self.weights.len()
    }

    fn predict(
        &self,
        predictions: &mut [f32],
        graph: &Graph,
        node_features: &[f32],
        dimension: usize,
    ) -> Result<(), String> {
        self.validate_features_for_prediction(predictions, graph, node_features, dimension)?;

        let method = self.edge_embedding_method_name.get_method();

        predictions
            .par_iter_mut()
            .zip(graph.par_iter_directed_edge_node_ids())
            .for_each(|(prediction, (_, src, dst))| {
                *prediction = unsafe {
                    self.get_unsafe_prediction(src, dst, node_features, dimension, method)
                        .1
                };
            });

        Ok(())
    }
}
