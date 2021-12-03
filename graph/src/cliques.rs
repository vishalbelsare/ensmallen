use super::*;
use indicatif::ProgressIterator;
use log::info;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Hash, Clone, Debug, PartialEq)]
pub struct Clique {
    graph: Graph,
    node_ids: Vec<NodeT>,
}

use std::string::ToString;
impl ToString for Clique {
    fn to_string(&self) -> String {
        let show_node_type = if self.graph.has_node_types() {
            unsafe {
                !self
                    .graph
                    .has_unchecked_isomorphic_node_types_from_node_ids(self.node_ids.as_ref())
            }
        } else {
            false
        };
        format!(
            concat!(
                "<p>",
                "Clique containing {nodes_number} nodes. ",
                "Specifically, the nodes involved in the clique are: {nodes}.",
                "{node_types_counts}",
                "{edge_types_counts}",
                "</p>",
            ),
            nodes_number = to_human_readable_high_integer(self.len() as usize),
            nodes = unsafe {
                get_unchecked_formatted_list(
                    &self
                        .node_ids
                        .iter()
                        .map(|&node_id| {
                            self.graph.get_unchecked_succinct_node_description(
                                node_id,
                                self.len(),
                                show_node_type,
                            )
                        })
                        .collect::<Vec<String>>(),
                    Some(5),
                )
            },
            node_types_counts = unsafe {
                self.graph
                    .get_unchecked_node_type_id_counts_hashmap_from_node_ids(self.node_ids.as_ref())
                    .map_or_else(
                        |_| "".to_string(),
                        |count| {
                            format!(
                                " Its nodes have {}.",
                                self.graph
                                    .get_unchecked_node_types_description_from_count(count)
                            )
                        },
                    )
            },
            edge_types_counts = unsafe {
                self.graph
                    .get_unchecked_edge_type_id_counts_hashmap_from_node_ids(self.node_ids.as_ref())
                    .map_or_else(
                        |_| "".to_string(),
                        |count| {
                            format!(
                                " Its edges have {}.",
                                self.graph
                                    .get_unchecked_edge_types_description_from_count(count)
                            )
                        },
                    )
            }
        )
    }
}

impl PartialOrd for Clique {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.len().cmp(&other.len()))
    }
}

impl Clique {
    pub(crate) fn from_node_ids(graph: &Graph, node_ids: Vec<NodeT>) -> Clique {
        Clique {
            graph: graph.clone(),
            node_ids,
        }
    }

    /// Return length of the Clique.
    pub fn len(&self) -> NodeT {
        self.node_ids.len() as NodeT
    }

    /// Return the node IDs of the nodes composing the clique.
    pub fn get_node_ids(&self) -> Vec<NodeT> {
        self.node_ids.clone()
    }

    /// Return the node names of the nodes composing the Clique.
    pub fn par_iter_node_names(&self) -> impl IndexedParallelIterator<Item = String> + '_ {
        self.node_ids.par_iter().map(move |&node_id| unsafe {
            self.graph.get_unchecked_node_name_from_node_id(node_id)
        })
    }

    /// Return the node names of the nodes composing the Clique.
    pub fn get_node_names(&self) -> Vec<String> {
        self.par_iter_node_names().collect()
    }
}

impl Graph {
    /// Returns parallel iterator over the graph cliques with at least `minimum_degree` nodes.
    ///
    /// # Arguments
    /// `minimum_degree`: Option<NodeT> - The optional minimum degree, by default 5.
    pub fn par_iter_cliques(
        &self,
        minimum_degree: Option<NodeT>,
    ) -> Result<impl Iterator<Item = Clique> + '_> {
        self.must_be_undirected()?;
        // First of all we set the minimum degree, which if None were provided is set to 5.
        let minimum_degree = minimum_degree.unwrap_or(5);
        // We create a vector with the initial node degrees of the graph, wrapped into atomic.
        info!("Create the node degrees vector.");
        let mut node_degrees: Vec<AtomicU32> = Vec::with_capacity(self.get_nodes_number() as usize);
        self.par_iter_node_degrees()
            .map(|degree| AtomicU32::new(degree))
            .collect_into_vec(&mut node_degrees);
        // We define the method we use to update the node degrees.
        let update_node_degree = |node_id: NodeT| {
            node_degrees[node_id as usize].store(0, Ordering::Relaxed);
            unsafe { self.iter_unchecked_neighbour_node_ids_from_source_node_id(node_id) }
                .for_each(|dst| {
                    node_degrees[dst as usize]
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |degree| {
                            Some(degree.saturating_sub(1))
                        })
                        .unwrap();
                });
        };
        // We do a preliminary filtering round removing the nodes with degree
        // lower than the provided minimum amount.
        // This initial filtering is also done again afterwards, but
        // by doing this preliminarly with an iterator we can allocate
        // a much smaller vector of nodes of interest.
        info!("Preliminary degree-based filtering.");
        let mut node_ids = self
            .par_iter_node_ids()
            .filter_map(|node_id| {
                // We retrieve the current node degree.
                let degree = node_degrees[node_id as usize].load(Ordering::Relaxed);
                // If the degree is zero, we have already dropped this node from the game.
                if degree < minimum_degree {
                    if degree > 0 {
                        update_node_degree(node_id);
                    }
                    None
                } else {
                    Some(node_id)
                }
            })
            .collect::<Vec<NodeT>>();
        // Compute a small report.
        let removed_nodes_number = self.get_nodes_number() as usize - node_ids.len();
        info!(
            concat!(
                "The preliminary filtering has removed ",
                "{removed_nodes_number} nodes ({percentage:.2})."
            ),
            removed_nodes_number = to_human_readable_high_integer(removed_nodes_number),
            percentage = removed_nodes_number as f64 / self.get_nodes_number() as f64 * 100.0
        );
        // Start to iterate over the node degrees vector
        // and in every iteration remove the nodes with degree smaller than
        // the provided amount. We iterate until no edit operation is
        // done.
        info!("Start iterations.");
        let mut current_iteration = 0;
        let minimum_degree_minus_one = (minimum_degree - 1) as usize;
        loop {
            let previous_nodes_number = node_ids.len();
            node_ids = node_ids.into_par_iter()
                .filter_map(|node_id| {
                    // We retrieve the current node degree.
                    let degree = node_degrees[node_id as usize].load(Ordering::Relaxed);
                    // If the degree is higher than the minimum degree
                    if degree > minimum_degree {
                        // If we are in a node that has, currently, degree higher or equal
                        // than the minimum degree, we check if is NOT inside a clique
                        // of size at least equal to the provided minimum degree.
                        let neighbours = unsafe {
                            self.iter_unchecked_unique_neighbour_node_ids_from_source_node_id(node_id)
                        }
                        .filter(|&dst| {
                            node_degrees[dst as usize].load(Ordering::Relaxed) >= minimum_degree
                        })
                        .collect::<Vec<NodeT>>();

                        // To check that the node may be part of a clique of size at least `minimum_degree`,
                        // then at least `minimum_degree - 1` neighbouring nodes must have at least `minimum_degree - 1`
                        // neighbouring nodes in the set of nodes of the initial node.
                        if neighbours
                            .iter()
                            .filter(|&&neighbour_node_id| {
                                iter_set::intersection(
                                    neighbours.iter().cloned(),
                                    unsafe{self.iter_unchecked_unique_neighbour_node_ids_from_source_node_id(
                                        neighbour_node_id,
                                    )},
                                )
                                .take(minimum_degree_minus_one)
                                .count()
                                    == minimum_degree_minus_one
                            })
                            .take(minimum_degree_minus_one)
                            .count()
                            == minimum_degree_minus_one
                        {
                            // We keep the node.
                            return Some(node_id);
                        }
                    }
                    update_node_degree(node_id);
                    None
                })
                .collect::<Vec<NodeT>>();
            // We check if we already have to stop.
            let currently_removed_nodes = previous_nodes_number - node_ids.len();
            if currently_removed_nodes == 0 {
                break;
            }
            info!(
                "#{current_iteration}: removed {currently_removed_nodes} nodes, remaining {remaining_nodes} ({percentage:.2}%).",
                current_iteration=current_iteration,
                currently_removed_nodes=to_human_readable_high_integer(currently_removed_nodes),
                remaining_nodes=to_human_readable_high_integer(node_ids.len()),
                percentage = node_ids.len() as f64 / self.get_nodes_number() as f64 * 100.0
            );
            current_iteration += 1;
        }
        info!(
            "Searching cliques in the remaining {} nodes",
            to_human_readable_high_integer(node_ids.len())
        );
        // The next step in our algorithm will be to collapse the nodes
        // that, after having removed all the low centrality nodes,
        // are now connected isomorphic nodes. We collapse these nodes
        // into "virtual" hyper-nodes.
        let neighbours_hashes = node_ids
            .par_iter()
            .map(|&node_id| unsafe {
                let mut seed: u64 = 0xDEADBEEFC0FEBABE_u64
                    .wrapping_mul(node_degrees[node_id as usize].load(Ordering::Relaxed) as u64);
                let mut added_selfloop = false;
                for neighbour_id in self
                    .iter_unchecked_unique_neighbour_node_ids_from_source_node_id(node_id)
                    .filter(|&dst| node_degrees[dst as usize].load(Ordering::Relaxed) > 0)
                    .take(100)
                    .map(|neighbour_id| neighbour_id as u64)
                {
                    if !added_selfloop && neighbour_id > node_id as u64 {
                        added_selfloop = true;
                        seed = (seed ^ node_id as u64).wrapping_add(0x0A2126967AE81C95)
                    }
                    seed = (seed ^ neighbour_id).wrapping_add(0x0A2126967AE81C95)
                }
                seed
            })
            .collect::<Vec<u64>>();
        // Get the indices of the filtered hashes for sorting
        let mut indices = (0..(neighbours_hashes.len() as NodeT)).collect::<Vec<NodeT>>();
        // Then we sort it according to the hash and the node ID.
        indices.par_sort_unstable_by(|&a, &b| {
            match neighbours_hashes[a as usize].cmp(&neighbours_hashes[b as usize]) {
                std::cmp::Ordering::Equal => node_ids[a as usize].cmp(&node_ids[b as usize]),
                std::cmp::Ordering::Greater => std::cmp::Ordering::Greater,
                std::cmp::Ordering::Less => std::cmp::Ordering::Less,
            }
        });
        let number_of_nodes = indices.len();
        // We compute the isomorphic groups
        let isomorphic_groups =
            (0..number_of_nodes.saturating_sub(1))
                .into_par_iter()
                .filter_map( |i| unsafe {
                    let index = indices[i];
                    let node_id = node_ids[index as usize];
                    let node_hash = neighbours_hashes[index as usize];
                    if i != 0 && node_hash == neighbours_hashes[indices[i - 1] as usize]
                        || node_hash != neighbours_hashes[indices[i + 1] as usize]
                    {
                        return None;
                    }
                    let mut candidate_isomorphic_groups = vec![vec![node_id]];
                    for other_node_id in ((i + 1)..number_of_nodes)
                        .take_while(|&j| node_hash == neighbours_hashes[indices[j] as usize])
                        .map(|j| node_ids[indices[j] as usize])
                    {
                        if let Some(isomorphic_group) = candidate_isomorphic_groups.iter_mut().find(
                            |candidate_isomorphic_group| {
                                let node_id = candidate_isomorphic_group[0];
                                self.iter_unchecked_unique_neighbour_node_ids_from_source_node_id(
                                node_id,
                            )
                            .filter(|&dst| {
                                dst != other_node_id &&
                                node_degrees[dst as usize].load(Ordering::Relaxed) > 0
                            })
                            .zip(
                                self.iter_unchecked_unique_neighbour_node_ids_from_source_node_id(
                                    other_node_id,
                                )
                                .filter(|&dst| {
                                    dst != node_id &&
                                    node_degrees[dst as usize].load(Ordering::Relaxed) > 0
                                }),
                            )
                            .all(
                                |(first_node_neighbour_id, second_node_neighbour_id)| {
                                    first_node_neighbour_id == second_node_neighbour_id
                                },
                            )
                            },
                        ) {
                            // We add the current node to the isomorphic group
                            isomorphic_group.push(other_node_id);
                        } else {
                            // We begin another isomorphic group
                            candidate_isomorphic_groups.push(vec![other_node_id]);
                        }
                    }
                    // We need to check that the isomorphic groups
                    // are not neither tautological nor
                    candidate_isomorphic_groups = candidate_isomorphic_groups
                        .into_iter()
                        .filter_map(|candidate_isomorphic_group| {
                            if candidate_isomorphic_group.len() > 1 {
                                let root_node_id = candidate_isomorphic_group[0];
                                Some(iter_set::intersection(
                                    candidate_isomorphic_group.into_iter().skip(1),
                                    self.iter_unchecked_unique_neighbour_node_ids_from_source_node_id(
                                        root_node_id,
                                    ).map(|node_id| {
                                        // We mark this node as with no degree, since it will
                                        // be functionally merged with the root node of this
                                        // isomorphic node group.
                                        node_degrees[node_id as usize].store(0, Ordering::Relaxed);
                                        node_id
                                    }),
                                ).chain(vec![root_node_id].into_iter()).collect::<Vec<NodeT>>())
                            } else {
                                None
                            }
                        })
                        .filter(|candidate_isomorphic_group| candidate_isomorphic_group.len() > 1)
                        .collect();
                    if candidate_isomorphic_groups.is_empty() {
                        None
                    } else {
                        Some(candidate_isomorphic_groups)
                    }
                })
                .flat_map(|candidate_isomorphic_groups| candidate_isomorphic_groups)
                .map(|mut candidate_isomorphic_group| {
                    let root_node = candidate_isomorphic_group.pop().unwrap();
                    (root_node, candidate_isomorphic_group)
                })
                .collect::<HashMap<NodeT, Vec<NodeT>>>();
        // Creating another info log.
        info!(
            "Identified {} isomorphic groups.",
            to_human_readable_high_integer(isomorphic_groups.len())
        );

        //===========================================
        // Start computation of dominating set.
        //===========================================
        info!("Computing dominating set.");
        // We convert the atomic degrees to non-atomic.
        let mut node_degrees =
            unsafe { std::mem::transmute::<Vec<AtomicU32>, Vec<NodeT>>(node_degrees) };
        let mut node_degrees_copy = node_degrees.clone();
        // Finally, we compute the dominating set of the nodes
        // and we obtain the set of nodes from where cliques may
        // be computed.
        let mut dominating_set = Vec::new();
        loop {
            let (node_id, degree) = node_degrees_copy.par_iter().cloned().argmax().unwrap();
            if degree == 0 {
                break;
            }
            dominating_set.push(node_id as NodeT);
            let covered_nodes = unsafe {
                self.iter_unchecked_unique_neighbour_node_ids_from_source_node_id(node_id as NodeT)
            }
            .filter(|&neighbour_node_id| node_degrees_copy[neighbour_node_id as usize] > 0)
            .collect::<Vec<NodeT>>();
            // We mark as covered the central node and all of its neighbours.
            node_degrees_copy[node_id] = 0;
            // Since the central node is covered, the degree of all of its
            // neighbours must be decreased by one.
            covered_nodes.iter().for_each(|&node_id| {
                node_degrees_copy[node_id as usize] -= 1;
            });
            let node_degrees_copy_atomic =
                unsafe { std::mem::transmute::<Vec<NodeT>, Vec<AtomicU32>>(node_degrees_copy) };
            covered_nodes
                .into_par_iter()
                .for_each(|neighbour_node_id| unsafe {
                    self.iter_unchecked_unique_neighbour_node_ids_from_source_node_id(
                        neighbour_node_id,
                    )
                    .for_each(|target_node_id| {
                        node_degrees_copy_atomic[target_node_id as usize]
                            .fetch_update(Ordering::Acquire, Ordering::Acquire, |degree| {
                                Some(degree.saturating_sub(1))
                            })
                            .unwrap();
                    });
                });
            node_degrees_copy = unsafe {
                std::mem::transmute::<Vec<AtomicU32>, Vec<NodeT>>(node_degrees_copy_atomic)
            };
        }

        info!(
            "Dominating set with {} nodes.",
            to_human_readable_high_integer(dominating_set.len())
        );

        let pb = get_loading_bar(true, "Computing cliques", dominating_set.len());

        // Actually compute and return cliques.
        Ok(dominating_set
            .into_iter()
            .progress_with(pb)
            .filter_map(move |node_id| {
                // First of all, we check if this node is a root of isomorphic nodes
                // and whether it has a degree equal to the number of nodes in the isomorphic
                // group: if this is true, the maximum clique of this node is equal to
                // the isomorphic group.
                if let Some(isomorphic_group) = isomorphic_groups.get(&node_id) {
                    if isomorphic_group.len() == node_degrees[node_id as usize] as usize {
                        let mut clique = isomorphic_group.clone();
                        clique.push(node_id);
                        return Some(vec![clique]);
                    }
                }
                if node_degrees[node_id as usize] < minimum_degree {
                    return None;
                }
                // Otherwise we need to check whether this node is in a clique.
                let neighbours = unsafe {
                    self.iter_unchecked_unique_neighbour_node_ids_from_source_node_id(node_id)
                }
                .filter(|&dst| node_degrees[dst as usize] >= minimum_degree)
                .collect::<Vec<NodeT>>();
                // If in the meantime its neighbours were removed, we can skip this node.
                if neighbours.is_empty() {
                    return None;
                }
                // The cliques vectors starts out as the root node and the first
                // neighbour of the root node.
                let mut clique = vec![node_id, neighbours[0]];
                clique.sort_unstable();
                let mut cliques: Vec<Vec<NodeT>> = vec![clique];
                // We iterate over the neighbours
                neighbours.into_iter().skip(1).for_each(|inner_node_id| {
                    // If the current neighbour wont fit in any of the other cliques
                    // we need to prepare a vector where to store its current matches.
                    let mut possible_new_cliques = Vec::new();
                    // We start to iterate over the existing growing cliques.
                    let number_of_matches = cliques
                        .iter_mut()
                        .map(|clique| unsafe {
                            // We count the number of matches in the current clique.
                            let matches = iter_set::intersection(
                                clique.iter().cloned(),
                                self.iter_unchecked_unique_neighbour_node_ids_from_source_node_id(
                                    inner_node_id,
                                ),
                            )
                            .collect::<Vec<NodeT>>();
                            // If we have a perfect match we can add the current
                            // node to this clique. Note that we cannot stop
                            // at this point, as the node may be shared between
                            // multiple cliques.
                            if matches.len() == clique.len() {
                                // Reduce the degree of the current node and all of the
                                // other nodes in the clique.
                                node_degrees[inner_node_id as usize] -= clique.len() as NodeT;
                                clique.iter().for_each(|&clique_node_id| {
                                    node_degrees[clique_node_id as usize] -= 1;
                                });
                                clique.push(inner_node_id);
                                clique.sort_unstable();
                                1
                            // Otherwise if the match is not perfect but we still
                            // have some matches we need to store these matches
                            // in the new clique we are growing for this node.
                            } else if matches.len() > 0 {
                                node_degrees[inner_node_id as usize] -= matches.len() as NodeT;
                                matches.iter().for_each(|&clique_node_id| {
                                    node_degrees[clique_node_id as usize] -= 1;
                                });
                                possible_new_cliques.push(matches);
                                0
                            } else {
                                0
                            }
                        })
                        .sum::<usize>();
                    // If the total number of matches is zero
                    if number_of_matches == 0 {
                        // We add the current node to the currently growing clique
                        if possible_new_cliques.is_empty() {
                            possible_new_cliques.push(Vec::new());
                        }
                        possible_new_cliques.iter_mut().for_each(|clique| {
                            clique.push(inner_node_id);
                            clique.sort_unstable();
                        });
                        // and push the clique to the set of cliques.
                        cliques.extend(possible_new_cliques);
                    }
                });
                Some(
                    cliques
                        .into_iter()
                        .map(|clique| {
                            clique
                                .into_iter()
                                .flat_map(|node_id| {
                                    if let Some(isomorphic_group) = isomorphic_groups.get(&node_id)
                                    {
                                        isomorphic_group.clone()
                                    } else {
                                        vec![node_id]
                                    }
                                })
                                .collect::<Vec<NodeT>>()
                        })
                        .filter(|clique| clique.len() > minimum_degree as usize)
                        .collect::<Vec<Vec<NodeT>>>(),
                )
            })
            .flat_map(move |cliques| {
                cliques
                    .into_iter()
                    .map(move |clique| Clique::from_node_ids(self, clique))
            }))
    }

    /// Returns graph cliques with at least `minimum_degree` nodes.
    ///
    /// # Arguments
    /// `minimum_degree`: Option<NodeT> - The optional minimum degree, by default 5.
    pub fn get_cliques(&self, minimum_degree: Option<NodeT>) -> Result<Vec<Clique>> {
        Ok(self.par_iter_cliques(minimum_degree)?.collect())
    }

    /// Returns number of graph cliques with at least `minimum_degree` nodes.
    ///
    /// # Arguments
    /// `minimum_degree`: Option<NodeT> - The optional minimum degree, by default 5.
    pub fn get_cliques_number(&self, minimum_degree: Option<NodeT>) -> Result<usize> {
        Ok(self.par_iter_cliques(minimum_degree)?.count())
    }
}
