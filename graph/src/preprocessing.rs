use super::*;
use indicatif::ProgressIterator;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;
use vec_rand::gen_random_vec;

/// Return training batches for Word2Vec models.
///
/// The batch is composed of a tuple as the following:
///
/// - (Contexts indices, central nodes indices): the tuple of nodes
///
/// This does not provide any output value as the model uses NCE loss
/// and basically the central nodes that are fed as inputs work as the
/// outputs value.
///
/// # Arguments
///
/// * sequences: Vec<Vec<usize>> - the sequence of sequences of integers to preprocess.
/// * window_size: usize - Window size to consider for the sequences.
///
pub fn word2vec<'a>(
    sequences: impl ParallelIterator<Item = Vec<NodeT>> + 'a,
    window_size: usize,
) -> Result<(Vec<Vec<NodeT>>, Vec<NodeT>), String> {
    let context_length = window_size.checked_mul(2).ok_or(
        "The given window size is too big, using this would result in an overflowing of a u64.",
    )?;

    Ok(sequences
        .flat_map_iter(|sequence| {
            sequence
                .iter()
                .enumerate()
                .filter_map(|(i, word)| {
                    let start = if i <= window_size { 0 } else { i - window_size };
                    let end = min!(sequence.len(), i + window_size);
                    if end - start == context_length {
                        Some((sequence[start..end].to_vec(), *word))
                    } else {
                        None
                    }
                })
                .collect::<Vec<(Vec<NodeT>, NodeT)>>()
        })
        .unzip())
}

/// Return triple with CSR representation of cooccurrence matrix.
///
/// The first vector has the sources, the second vector the destinations
/// and the third one contains the min-max normalized frequencies.
///
/// # Arguments
///
/// * sequences:Vec<Vec<usize>> - the sequence of sequences of integers to preprocess.
/// * window_size: Option<usize> - Window size to consider for the sequences.
/// * verbose: Option<bool>,
///     Wethever to show the progress bars.
///     The default behaviour is false.
///     
pub fn cooccurence_matrix(
    sequences: impl ParallelIterator<Item = Vec<NodeT>>,
    window_size: usize,
    number_of_sequences: usize,
    verbose: bool,
) -> Result<(Words, Words, Frequencies), String> {
    let mut cooccurence_matrix: HashMap<(NodeT, NodeT), f64> = HashMap::new();
    let pb1 = get_loading_bar(verbose, "Computing frequencies", number_of_sequences);
    let vec = sequences.collect::<Vec<Vec<NodeT>>>();
    vec.iter().progress_with(pb1).for_each(|sequence| {
        let walk_length = sequence.len();
        for (central_index, &central_word_id) in sequence.iter().enumerate() {
            for distance in 1..1 + window_size {
                if central_index + distance >= walk_length {
                    break;
                }
                let context_id = sequence[central_index + distance];
                if central_word_id < context_id {
                    *cooccurence_matrix
                        .entry((central_word_id as NodeT, context_id as NodeT))
                        .or_insert(0.0) += 1.0 / distance as f64;
                } else {
                    *cooccurence_matrix
                        .entry((context_id as NodeT, central_word_id as NodeT))
                        .or_insert(0.0) += 1.0 / distance as f64;
                }
            }
        }
    });

    let elements = cooccurence_matrix.len() * 2;
    let mut max_frequency = 0.0;
    let mut words: Vec<NodeT> = vec![0; elements];
    let mut contexts: Vec<NodeT> = vec![0; elements];
    let mut frequencies: Vec<f64> = vec![0.0; elements];
    let pb2 = get_loading_bar(
        verbose,
        "Converting mapping into CSR matrix",
        cooccurence_matrix.len(),
    );

    cooccurence_matrix
        .iter()
        .progress_with(pb2)
        .enumerate()
        .for_each(|(i, ((word, context), frequency))| {
            let (k, j) = (i * 2, i * 2 + 1);
            if *frequency > max_frequency {
                max_frequency = *frequency;
            }
            words[k] = *word;
            words[j] = words[k];
            contexts[k] = *context;
            contexts[j] = contexts[k];
            frequencies[k] = *frequency;
            frequencies[j] = frequencies[k];
        });

    frequencies
        .par_iter_mut()
        .for_each(|frequency| *frequency /= max_frequency);

    Ok((words, contexts, frequencies))
}

/// # Preprocessing for ML algorithms on graph.
impl Graph {
    /// Return training batches for Node2Vec models.
    ///
    /// The batch is composed of a tuple as the following:
    ///
    /// - (Contexts indices, central nodes indices): the tuple of nodes
    ///
    /// This does not provide any output value as the model uses NCE loss
    /// and basically the central nodes that are fed as inputs work as the
    /// outputs value.
    ///
    /// # Arguments
    ///
    /// * walk_parameters: &WalksParameters - the weighted walks parameters.
    /// * quantity: usize - Number of nodes to consider.
    /// * window_size: usize - Window size to consider for the sequences.
    ///
    pub fn node2vec(
        &self,
        walk_parameters: &WalksParameters,
        quantity: NodeT,
        window_size: usize,
    ) -> Result<(Contexts, Words), String> {
        // do the walks and check the result
        let walks = self.random_walks_iter(quantity, walk_parameters)?;
        word2vec(walks, window_size)
    }

    /// Return triple with CSR representation of cooccurrence matrix.
    ///
    /// The first vector has the sources, the second vector the destinations
    /// and the third one contains the min-max normalized frequencies.
    ///
    /// # Arguments
    ///
    /// * parameters: &WalksParameters - the walks parameters.
    /// * window_size: Option<usize> - Window size to consider for the sequences.
    /// * verbose: Option<bool>,
    ///     Wethever to show the progress bars.
    ///     The default behaviour is false.
    ///     
    pub fn cooccurence_matrix(
        &self,
        walks_parameters: &WalksParameters,
        window_size: usize,
        verbose: bool,
    ) -> Result<(Words, Words, Frequencies), String> {
        let walks = self.complete_walks_iter(walks_parameters)?;
        cooccurence_matrix(
            walks,
            window_size,
            (self.get_unique_sources_number() * walks_parameters.iterations) as usize,
            verbose,
        )
    }

    /// Returns triple with source nodes, destination nodes and labels for training model for link prediction.
    ///
    /// # Arguments
    ///
    /// * idx:u64 - The index of the batch to generate, behaves like a random random_state,
    /// * batch_size:usize - The maximal size of the batch to generate,
    /// * negative_samples: f64 - The component of netagetive samples to use,
    /// * avoid_false_negatives: bool - Wether to remove the false negatives when generated.
    ///     - It should be left to false, as it has very limited impact on the training, but enabling this will slow things down.
    /// * maximal_sampling_attempts: usize - Number of attempts to execute to sample the negative edges.
    /// * graph_to_avoid: Option<&Graph> - The graph whose edges are to be avoided during the generation of false negatives,
    ///
    pub fn link_prediction(
        &self,
        idx: u64,
        batch_size: usize,
        negative_samples: f64,
        avoid_false_negatives: bool,
        maximal_sampling_attempts: usize,
        graph_to_avoid: Option<&Graph>,
    ) -> Result<(Contexts, Vec<bool>), String> {
        // xor the random_state with a constant so that we have a good amount of 0s and 1s in the number
        // even with low values (this is needed becasue the random_state 0 make xorshift return always 0)
        let random_state = idx ^ SEED_XOR as u64;

        if negative_samples < 0.0 || !negative_samples.is_finite() {
            return Err(String::from("Negative sample must be a posive real value."));
        }

        // The number of negatives is given by computing their fraction of batchsize
        let negatives_number: usize =
            ((batch_size as f64 / (1.0 + negative_samples)) * negative_samples) as usize;
        // All the remaining values then are positives
        let positives_number: usize = batch_size - negatives_number;
        let graph_has_no_self_loops = !self.has_selfloops();

        let edges_number = self.get_edges_number() as u64;
        let nodes_number = self.get_nodes_number() as u64;

        let mut rng: StdRng = SeedableRng::seed_from_u64(random_state);
        let mut indices: Vec<usize> = (0..batch_size).collect();
        indices.shuffle(&mut rng);

        let mut contexts = vec![vec![0; 2]; batch_size];
        let mut labels = vec![false; batch_size];

        gen_random_vec(positives_number, random_state)
            .iter()
            .enumerate()
            .for_each(|(i, sampled)| {
                let (src, dst) = self.get_edge_from_edge_id(sampled % edges_number);
                contexts[indices[i]][0] = src;
                contexts[indices[i]][1] = dst;
                labels[indices[i]] = true;
            });

        for (i, sampled) in gen_random_vec(negatives_number, random_state)
            .iter()
            .enumerate()
        {
            let mut attempts = 0;
            loop {
                if attempts > maximal_sampling_attempts {
                    return Err(format!(
                        concat!(
                            "Executed more than {} attempts to sample a negative edge. ",
                            "If your graph is so small that you see this error, you may want to consider ",
                            "using one of the edge embedding transformer from the Embiggen library."
                        ),
                        maximal_sampling_attempts
                    ));
                }
                attempts += 1;
                let random_src = sampled & 0xffffffff; // We need this to be an u64.
                let random_dst = sampled >> 32; // We need this to be an u64.
                                                // This technique is taken from:
                                                // https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
                let src = ((random_src * nodes_number) >> 32) as NodeT;
                let dst = ((random_dst * nodes_number) >> 32) as NodeT;
                if avoid_false_negatives && self.has_edge(src, dst, None) {
                    continue;
                }
                if let Some(g) = &graph_to_avoid {
                    if g.has_edge(src, dst, None) {
                        continue;
                    }
                }
                if graph_has_no_self_loops && src == dst {
                    continue;
                }
                contexts[indices[positives_number + i]][0] = src;
                contexts[indices[positives_number + i]][1] = dst;
                break;
            }
        }

        Ok((contexts, labels))
    }
}
