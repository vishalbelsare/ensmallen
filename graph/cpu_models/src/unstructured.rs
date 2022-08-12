use crate::*;
use express_measures::{normalize_vector_inplace, ThreadFloat};
use graph::{Graph, NodeT, ThreadDataRaceAware};
use rayon::prelude::*;
use num_traits::Coerced;
use vec_rand::splitmix64;

#[derive(Clone, Debug)]
pub struct Unstructured {
    model: BasicSiameseModel,
}

impl From<BasicSiameseModel> for Unstructured {
    fn from(model: BasicSiameseModel) -> Self {
        Self { model }
    }
}

impl GraphEmbedder for Unstructured {
    fn get_model_name(&self) -> String {
        "Unstructured".into()
    }

    fn get_random_state(&self) -> u64 {
        self.model.get_random_state()
    }

    fn is_verbose(&self) -> bool {
        self.model.is_verbose()
    }

    fn get_dtype(&self) -> String {
        self.model.get_dtype()
    }

    fn get_number_of_epochs(&self) -> usize {
        self.model.get_number_of_epochs()
    }

    fn get_embedding_shapes(&self, graph: &Graph) -> Result<Vec<MatrixShape>, String> {
        Ok(vec![(
            graph.get_number_of_nodes() as usize,
            self.model.get_embedding_size(),
        )
            .into()])
    }

    fn _fit_transform<F: Coerced<f32> + ThreadFloat>(
        &self,
        graph: &Graph,
        embedding: &mut [&mut [F]],
    ) -> Result<(), String> {
        let embedding_size = self.model.get_embedding_size();
        let scale_factor = (embedding_size as f32).sqrt();
        let mut learning_rate = self.model.get_learning_rate() / scale_factor;
        let mut random_state = self.model.get_random_state();

        let shared_node_embedding = ThreadDataRaceAware::new(&mut embedding[0]);

        // Depending whether verbosity was requested by the user
        // we create or not a visible progress bar to show the progress
        // in the training epochs.
        let pb = self.get_loading_bar();

        // We start to loop over the required amount of epochs.
        for _ in 0..self.get_number_of_epochs() {
            // We update the random state used to generate the random walks
            // and the negative samples.
            random_state = splitmix64(random_state);

            // We iterate over the graph edges.
            let total_variation = graph
                .par_iter_siamese_mini_batch(
                    random_state,
                    graph.get_number_of_directed_edges() as usize,
                )
                .map(|(_, src, dst, not_src, not_dst)| {
                    let src = src as usize;
                    let dst = dst as usize;
                    let not_src = not_src as usize;
                    let not_dst = not_dst as usize;
                    let src_embedding = unsafe {
                        &mut (*shared_node_embedding.get())
                            [(src * embedding_size)..((src + 1) * embedding_size)]
                    };
                    let not_src_embedding = unsafe {
                        &mut (*shared_node_embedding.get())
                            [(not_src * embedding_size)..((not_src + 1) * embedding_size)]
                    };
                    let dst_embedding = unsafe {
                        &mut (*shared_node_embedding.get())
                            [(dst * embedding_size)..((dst + 1) * embedding_size)]
                    };
                    let not_dst_embedding = unsafe {
                        &mut (*shared_node_embedding.get())
                            [(not_dst * embedding_size)..((not_dst + 1) * embedding_size)]
                    };

                    let (dst_norm, not_dst_norm, src_norm, not_src_norm) = (
                        normalize_vector_inplace(dst_embedding),
                        normalize_vector_inplace(not_dst_embedding),
                        normalize_vector_inplace(src_embedding),
                        normalize_vector_inplace(not_src_embedding),
                    );
                    let node_priors = get_node_priors(
                        graph,
                        &[
                            src as NodeT,
                            dst as NodeT,
                            not_src as NodeT,
                            not_dst as NodeT,
                        ],
                        learning_rate,
                    )
                    .into_iter()
                    .map(|prior| F::coerce_from(prior))
                    .collect::<Vec<F>>();

                    src_embedding
                        .iter_mut()
                        .zip(not_src_embedding.iter_mut())
                        .zip(dst_embedding.iter_mut().zip(not_dst_embedding.iter_mut()))
                        .map(
                            |((src_feature, not_src_feature), (dst_feature, not_dst_feature))| {
                                *src_feature /= src_norm;
                                *not_src_feature /= not_src_norm;
                                *dst_feature /= dst_norm;
                                *not_dst_feature /= not_dst_norm;

                                let positive_distance = *src_feature - *dst_feature;
                                let negative_distance = *not_src_feature - *not_dst_feature;
                                let loss = positive_distance.powf(F::coerce_from(2.0))
                                    - negative_distance.powf(F::coerce_from(2.0));

                                if loss > F::coerce_from(-self.model.relu_bias) {
                                    *src_feature -= positive_distance * node_priors[0];
                                    *dst_feature += positive_distance * node_priors[1];
                                    *not_src_feature += negative_distance * node_priors[2];
                                    *not_dst_feature -= negative_distance * node_priors[3];
                                }
                                loss.abs()
                            },
                        )
                        .sum::<F>()
                })
                .sum::<F>();
            if total_variation.is_zero() {
                break;
            }
            learning_rate *= self.model.get_learning_rate_decay();
            pb.inc(1);
            pb.set_message(format!(", variation: {:.4}", total_variation.coerce_into()));
        }
        Ok(())
    }
}
