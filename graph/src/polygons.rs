use super::*;
use bitvec::prelude::*;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

impl Graph {
    /// Returns number of triangles in the graph.
    ///
    /// # References
    /// This implementation is described in ["Faster Clustering Coefficient Using Vertex Covers"](https://ieeexplore.ieee.org/document/6693348).
    ///
    pub fn get_number_of_triangles(&self, verbose: Option<bool>) -> EdgeT {
        let verbose = verbose.unwrap_or(true);

        // First, we compute the set of nodes composing a vertex cover set.
        // This vertex cover is NOT minimal, but is a 2-approximation.
        let vertex_cover = self
            .get_approximated_vertex_cover(
                Some("decreasing_node_degree"),
                Some(true),
                Some(true),
                None,
            )
            .unwrap();

        let vertex_cover_size = vertex_cover.iter().filter(|cover| **cover).count();

        let pb = get_loading_bar(verbose, "Computing number of triangles", vertex_cover_size);

        let vertex_cover_reference = vertex_cover.as_slice();

        // We start iterating over the nodes in the cover using rayon to parallelize the procedure.
        vertex_cover
            .par_iter()
            .enumerate()
            .filter_map(|(first, is_cover)| {
                if *is_cover {
                    Some(first as NodeT)
                } else {
                    None
                }
            })
            .progress_with(pb)
            // For each node in the cover
            .flat_map(|first| {
                // We obtain the neighbours and collect them into a vector
                // We store them instead of using them in a stream because we will need
                // them multiple times below.
                let first_order_neighbours = unsafe {
                    self.edges
                        .get_unchecked_neighbours_node_ids_from_src_node_id(first)
                };

                let index = first_order_neighbours.partition_point(|&second| second < first);

                first_order_neighbours[..index]
                    .par_iter()
                    .filter_map(move |&second| {
                        if second != first && vertex_cover_reference[second as usize] {
                            Some((first, second, first_order_neighbours))
                        } else {
                            None
                        }
                    })
            })
            .map(|(first, second, first_order_neighbours)| {
                // We iterate over the neighbours
                // We compute the intersection of the neighbours.

                let mut first_neighbour_index = 0;
                let mut second_neighbour_index = 0;
                let mut partial_number_of_triangles: EdgeT = 0;

                let second_order_neighbours = unsafe {
                    self.edges
                        .get_unchecked_neighbours_node_ids_from_src_node_id(second)
                };

                while first_neighbour_index < first_order_neighbours.len()
                    && second_neighbour_index < second_order_neighbours.len()
                {
                    let first_order_neighbour = first_order_neighbours[first_neighbour_index];
                    // If this is a self-loop, we march on forward
                    if first_order_neighbour == second || first_order_neighbour == first {
                        first_neighbour_index += 1;
                        continue;
                    }
                    // If this is not an intersection, we march forward
                    let second_order_neighbour = second_order_neighbours[second_neighbour_index];
                    if first_order_neighbour < second_order_neighbour {
                        first_neighbour_index += 1;
                        continue;
                    }
                    if first_order_neighbour > second_order_neighbour {
                        second_neighbour_index += 1;
                        continue;
                    }
                    // If we reach here, we are in an intersection.
                    first_neighbour_index += 1;
                    second_neighbour_index += 1;
                    // If the inner node is as well in the vertex cover
                    // we only count this as one, as we will encounter
                    // combinations of these nodes multiple times
                    // while iterating the vertex cover nodes
                    partial_number_of_triangles +=
                        if vertex_cover_reference[first_order_neighbour as usize] {
                            1
                        } else {
                            // Otherwise we won't encounter again this
                            // node and we need to count the triangles
                            // three times.
                            3
                        };
                }
                partial_number_of_triangles
            })
            .sum::<EdgeT>()
    }

    /// Returns number of squares in the graph.
    ///
    /// # Arguments
    /// `verbose`: Option<bool> - Whether to show a loading bar. By default, True.
    ///
    /// # References
    /// This implementation is described in ["Faster Clustering Coefficient Using Vertex Covers"](https://ieeexplore.ieee.org/document/6693348).
    ///
    pub fn get_number_of_squares_tommy(&self, verbose: Option<bool>) -> EdgeT {
        // First, we compute the set of nodes composing a vertex cover set.
        // This vertex cover is NOT minimal, but is a 2-approximation.
        let vertex_cover = self
            .get_approximated_vertex_cover(
                Some("decreasing_node_degree"),
                Some(true),
                Some(true),
                None,
            )
            .unwrap();

        let vertex_cover_size = vertex_cover.iter().filter(|cover| **cover).count();

        let verbose = verbose.unwrap_or(true);

        let pb = get_loading_bar(verbose, "Computing number of squares", vertex_cover_size);

        let vertex_cover_reference = vertex_cover.as_slice();

        // We start iterating over the nodes in the cover using rayon to parallelize the procedure.
        vertex_cover
            .par_iter()
            .enumerate()
            .filter_map(|(first, is_cover)| {
                if *is_cover {
                    Some((first as NodeT, unsafe {
                        self.edges
                            .get_unchecked_neighbours_node_ids_from_src_node_id(first as NodeT)
                    }))
                } else {
                    None
                }
            })
            .progress_with(pb)
            .map(|(first, first_order_neighbours)| {
                let mut partial_squares_number = 0;

                for (i, &second) in first_order_neighbours.iter().enumerate() {
                    if vertex_cover_reference[second as usize] && second >= first {
                        continue;
                    }

                    let second_neighbours = unsafe {
                        self.edges
                            .get_unchecked_neighbours_node_ids_from_src_node_id(second as NodeT)
                    };

                    for &third in &first_order_neighbours[0..i] {
                        if third >= first {
                            break;
                        }
                        if vertex_cover_reference[third as usize] && third >= first {
                            continue;
                        }

                        let third_neighbours = unsafe {
                            self.edges
                                .get_unchecked_neighbours_node_ids_from_src_node_id(third as NodeT)
                        };

                        let mut second_neighbour_index = 0;
                        let mut third_neighbour_index = 0;

                        while second_neighbour_index < second_neighbours.len()
                            && third_neighbour_index < third_neighbours.len()
                        {
                            let second_neighbour = second_neighbours[second_neighbour_index];
                            // If this is a self-loop, we march on forward
                            if second_neighbour == third || second_neighbour == second {
                                second_neighbour_index += 1;
                                continue;
                            }
                            // If this is not an intersection, we march forward
                            let third_neighbour = third_neighbours[third_neighbour_index];
                            if second_neighbour < third_neighbour {
                                second_neighbour_index += 1;
                                continue;
                            }
                            if second_neighbour > third_neighbour {
                                third_neighbour_index += 1;
                                continue;
                            }
                            // If we reach here, we are in an intersection.
                            second_neighbour_index += 1;
                            third_neighbour_index += 1;

                            let fourth = second_neighbour;

                            if vertex_cover_reference[fourth as usize] && fourth >= first {
                                continue;
                            }
                            partial_squares_number += 1;
                        }
                    }
                }

                partial_squares_number
            })
            .sum::<EdgeT>()
    }

    /// Returns number of squares in the graph.
    ///
    /// # Arguments
    /// `verbose`: Option<bool> - Whether to show a loading bar. By default, True.
    ///
    /// # References
    /// This implementation is described in ["Faster Clustering Coefficient Using Vertex Covers"](https://ieeexplore.ieee.org/document/6693348).
    ///
    pub fn get_number_of_squares(&self, verbose: Option<bool>) -> EdgeT {
        // First, we compute the set of nodes composing a vertex cover set.
        // This vertex cover is NOT minimal, but is a 2-approximation.
        let vertex_cover = self
            .get_approximated_vertex_cover(
                Some("decreasing_node_degree"),
                Some(true),
                Some(true),
                None,
            )
            .unwrap();

        let vertex_cover_size = vertex_cover.iter().filter(|cover| **cover).count();

        let verbose = verbose.unwrap_or(true);

        let pb = get_loading_bar(verbose, "Computing number of squares", vertex_cover_size);

        let bitvecs = ThreadDataRaceAware::new(
            (0..rayon::current_num_threads())
                .map(|_| bitvec![u64, Lsb0; 0; self.get_number_of_nodes() as usize])
                .collect::<Vec<_>>(),
        );

        let vertex_cover_reference = vertex_cover.as_slice();

        // We start iterating over the nodes in the cover using rayon to parallelize the procedure.
        vertex_cover
            .par_iter()
            .enumerate()
            .filter_map(|(first, is_cover)| {
                if *is_cover {
                    Some((first as NodeT, unsafe {
                        self.edges
                            .get_unchecked_neighbours_node_ids_from_src_node_id(first as NodeT)
                    }))
                } else {
                    None
                }
            })
            .progress_with(pb)
            .map(|(first, first_order_neighbours)|{
                let thread_id = rayon::current_thread_index().expect("current_thread_id not called from a rayon thread. This should not be possible because this is in a Rayon Thread Pool.");
                let bitvec = unsafe{&mut (*bitvecs.get())[thread_id]};
                let mut partial_squares_number = 0;
                bitvec.fill(false);

                for &second in first_order_neighbours {
                    let second_order_neighbours = unsafe{self.edges
                        .get_unchecked_neighbours_node_ids_from_src_node_id(second as NodeT)};
                    for &third in second_order_neighbours {
                        if third >= first {
                            break;
                        }
                        if !vertex_cover_reference[third as usize] {
                            continue;
                        }
                        if unsafe{bitvec.replace_unchecked(third as usize, true)} {
                            continue;
                        }

                        let third_order_neighbours = unsafe{self.edges
                            .get_unchecked_neighbours_node_ids_from_src_node_id(third as NodeT)};
                        let mut first_neighbour_index = 0;
                        let mut third_neighbour_index = 0;
                        let mut in_vertex_cover: EdgeT = 0;
                        let mut not_in_vertex_cover: EdgeT = 0;

                        while first_neighbour_index < first_order_neighbours.len()
                            && third_neighbour_index < third_order_neighbours.len()
                        {
                            let first_order_neighbour =
                                first_order_neighbours[first_neighbour_index];
                            // If this is a self-loop, we march on forward
                            if first_order_neighbour == third || first_order_neighbour == first {
                                first_neighbour_index += 1;
                                continue;
                            }
                            // If this is not an intersection, we march forward
                            let third_order_neighbour =
                                third_order_neighbours[third_neighbour_index];
                            if first_order_neighbour < third_order_neighbour {
                                first_neighbour_index += 1;
                                continue;
                            }
                            if first_order_neighbour > third_order_neighbour {
                                third_neighbour_index += 1;
                                continue;
                            }
                            // If we reach here, we are in an intersection.
                            first_neighbour_index += 1;
                            third_neighbour_index += 1;

                            let forth = first_order_neighbour;

                            if vertex_cover_reference[forth as usize] {
                                in_vertex_cover += 1;
                            } else {
                                not_in_vertex_cover += 1;
                            };
                        }
                        partial_squares_number += (in_vertex_cover + not_in_vertex_cover)
                            * (in_vertex_cover + not_in_vertex_cover).saturating_sub(1)
                            + not_in_vertex_cover * not_in_vertex_cover.saturating_sub(1)
                            + 2 * not_in_vertex_cover * in_vertex_cover;
                    }
                }
                
                partial_squares_number
            }).sum::<EdgeT>() / 4
    }


    /// Returns number of squares in the graph.
    ///
    /// # Arguments
    /// `verbose`: Option<bool> - Whether to show a loading bar. By default, True.
    ///
    /// # References
    /// This implementation is described in ["Faster Clustering Coefficient Using Vertex Covers"](https://ieeexplore.ieee.org/document/6693348).
    ///
    pub fn get_number_of_squares_per_node(&self, verbose: Option<bool>) -> Vec<EdgeT> {
        // First, we compute the set of nodes composing a vertex cover set.
        // This vertex cover is NOT minimal, but is a 2-approximation.
        let vertex_cover = self
            .get_approximated_vertex_cover(
                Some("decreasing_node_degree"),
                Some(true),
                Some(true),
                None,
            )
            .unwrap();

        let vertex_cover_size = vertex_cover.iter().filter(|cover| **cover).count();

        let node_squares_number = unsafe {
            std::mem::transmute::<Vec<EdgeT>, Vec<AtomicU64>>(vec![
                0;
                self.get_number_of_nodes()
                    as usize
            ])
        };

        let verbose = verbose.unwrap_or(true);

        let pb = get_loading_bar(verbose, "Computing number of squares", vertex_cover_size);

        let vertex_cover_reference = vertex_cover.as_slice();

        // We start iterating over the nodes in the cover using rayon to parallelize the procedure.
        vertex_cover
            .par_iter()
            .enumerate()
            .filter_map(|(first, is_cover)| {
                if *is_cover {
                    Some((first as NodeT, unsafe {
                        self.edges
                            .get_unchecked_neighbours_node_ids_from_src_node_id(first as NodeT)
                    }))
                } else {
                    None
                }
            })
            .progress_with(pb)
            .for_each(|(first, first_order_neighbours)| {
                let mut first_squares = 0;
                for (i, &second) in first_order_neighbours.iter().enumerate() {
                    if vertex_cover_reference[second as usize] && second >= first {
                        continue;
                    }

                    let mut second_squares = 0;

                    let second_neighbours = unsafe {
                        self.edges
                            .get_unchecked_neighbours_node_ids_from_src_node_id(second as NodeT)
                    };

                    for &third in &first_order_neighbours[0..i] {
                        if third >= first {
                            break;
                        }
                        if vertex_cover_reference[third as usize] && third >= first {
                            continue;
                        }

                        let mut third_squares = 0;

                        let third_neighbours = unsafe {
                            self.edges
                                .get_unchecked_neighbours_node_ids_from_src_node_id(third as NodeT)
                        };

                        let mut second_neighbour_index = 0;
                        let mut third_neighbour_index = 0;

                        while second_neighbour_index < second_neighbours.len()
                            && third_neighbour_index < third_neighbours.len()
                        {
                            let second_neighbour = second_neighbours[second_neighbour_index];
                            // If this is a self-loop, we march on forward
                            if second_neighbour == third || second_neighbour == second {
                                second_neighbour_index += 1;
                                continue;
                            }
                            // If this is not an intersection, we march forward
                            let third_neighbour = third_neighbours[third_neighbour_index];
                            if second_neighbour < third_neighbour {
                                second_neighbour_index += 1;
                                continue;
                            }
                            if second_neighbour > third_neighbour {
                                third_neighbour_index += 1;
                                continue;
                            }
                            // If we reach here, we are in an intersection.
                            second_neighbour_index += 1;
                            third_neighbour_index += 1;

                            let fourth = second_neighbour;

                            if vertex_cover_reference[fourth as usize] && fourth >= first {
                                continue;
                            }
                            node_squares_number[fourth as usize].fetch_add(1, Ordering::Relaxed);
                            first_squares += 1;
                            second_squares += 1;
                            third_squares += 1;
                        }
                        node_squares_number[third as usize]
                            .fetch_add(third_squares, Ordering::Relaxed);
                    }
                    node_squares_number[second as usize]
                        .fetch_add(second_squares, Ordering::Relaxed);
                }
                node_squares_number[first as usize].fetch_add(first_squares, Ordering::Relaxed);
            });
        unsafe { std::mem::transmute::<Vec<AtomicU64>, Vec<EdgeT>>(node_squares_number) }
    }

    /// Returns total number of triads in the graph without taking into account weights.
    pub fn get_number_of_triads(&self) -> EdgeT {
        self.par_iter_node_degrees()
            .map(|degree| (degree as EdgeT) * (degree.saturating_sub(1) as EdgeT))
            .sum()
    }

    /// Returns total number of triads in the weighted graph.
    pub fn get_number_of_weighted_triads(&self) -> Result<f64> {
        Ok(self
            .par_iter_weighted_node_degrees()?
            .map(|degree| {
                if degree > 1.0 {
                    degree * (degree - 1.0)
                } else {
                    0.0
                }
            })
            .sum())
    }

    /// Returns transitivity of the graph without taking into account weights.
    ///
    /// # Arguments
    /// * `verbose`: Option<bool> - Whether to show a loading bar.
    pub fn get_transitivity(&self, verbose: Option<bool>) -> f64 {
        self.get_number_of_triangles(verbose) as f64 / self.get_number_of_triads() as f64
    }

    /// Returns number of triangles for all nodes in the graph.
    ///
    /// # Arguments
    /// * `verbose`: Option<bool> - Whether to show a loading bar.
    ///
    /// # References
    /// This implementation is described in ["Faster Clustering Coefficient Using Vertex Covers"](https://ieeexplore.ieee.org/document/6693348).
    ///
    /// # Safety
    /// This method does not support directed graphs and will raise a panic.
    /// It should automatically dispatched the naive version for these cases.
    pub fn get_number_of_triangles_per_node(&self, verbose: Option<bool>) -> Vec<EdgeT> {
        let node_triangles_number = unsafe {
            std::mem::transmute::<Vec<EdgeT>, Vec<AtomicU64>>(vec![
                0;
                self.get_number_of_nodes()
                    as usize
            ])
        };
        let verbose = verbose.unwrap_or(true);
        let vertex_cover = self
            .get_approximated_vertex_cover(None, None, None, None)
            .unwrap();
        let cover_size = vertex_cover
            .par_iter()
            .filter(|&&is_cover| is_cover)
            .count();
        let pb = get_loading_bar(
            verbose,
            "Computing number of triangles per node",
            cover_size,
        );

        let vertex_cover_reference = vertex_cover.as_slice();

        // We start iterating over the nodes in the cover using rayon to parallelize the procedure.
        vertex_cover
            .par_iter()
            .enumerate()
            .filter_map(|(node_id, is_cover)| {
                if *is_cover {
                    Some(node_id as NodeT)
                } else {
                    None
                }
            })
            .progress_with(pb)
            // For each node in the cover
            .flat_map(|first| {
                // We obtain the neighbours and collect them into a vector
                // We store them instead of using them in a stream because we will need
                // them multiple times below.
                let first_order_neighbours = unsafe {
                    self.edges
                        .get_unchecked_neighbours_node_ids_from_src_node_id(first)
                };

                let index = first_order_neighbours.partition_point(|&second| second < first);

                first_order_neighbours[..index]
                    .par_iter()
                    .filter_map(move |&second| {
                        if second != first && vertex_cover_reference[second as usize] {
                            Some((first, second, first_order_neighbours))
                        } else {
                            None
                        }
                    })
            })
            .for_each(|(first, second, first_order_neighbours)| {
                // We iterate over the neighbours
                // We compute the intersection of the neighbours.

                let mut first_neighbour_index = 0;
                let mut second_neighbour_index = 0;

                let second_order_neighbours = unsafe {
                    self.edges
                        .get_unchecked_neighbours_node_ids_from_src_node_id(second)
                };

                let mut first_triangles = 0;
                let mut second_triangles = 0;

                while first_neighbour_index < first_order_neighbours.len()
                    && second_neighbour_index < second_order_neighbours.len()
                {
                    let first_order_neighbour = first_order_neighbours[first_neighbour_index];
                    // If this is a self-loop, we march on forward
                    if first_order_neighbour == first || first_order_neighbour == second {
                        first_neighbour_index += 1;
                        continue;
                    }
                    // If this is not an intersection, we march forward
                    let second_order_neighbour = second_order_neighbours[second_neighbour_index];
                    if first_order_neighbour < second_order_neighbour {
                        first_neighbour_index += 1;
                        continue;
                    }
                    if first_order_neighbour > second_order_neighbour {
                        second_neighbour_index += 1;
                        continue;
                    }
                    // If we reach here, we are in an intersection.
                    first_neighbour_index += 1;
                    second_neighbour_index += 1;

                    let third = first_neighbour_index;

                    // If the inner node is as well in the vertex cover
                    // we only count this as one, as we will encounter
                    // combinations of these nodes multiple times
                    // while iterating the vertex cover nodes
                    first_triangles += 1;
                    if !vertex_cover_reference[second_order_neighbour as usize] {
                        // Otherwise we won't encounter again this
                        // node and we need to count the triangles
                        // three times.
                        second_triangles += 1;
                        node_triangles_number[third as usize].fetch_add(1, Ordering::Relaxed);
                    }
                }
                node_triangles_number[first as usize].fetch_add(first_triangles, Ordering::Relaxed);
                node_triangles_number[second as usize]
                    .fetch_add(second_triangles, Ordering::Relaxed);
            });

        unsafe { std::mem::transmute::<Vec<AtomicU64>, Vec<EdgeT>>(node_triangles_number) }
    }

    /// Returns iterator over the clustering coefficients for all nodes in the graph.
    ///
    /// # Arguments
    /// * `verbose`: Option<bool> - Whether to show a loading bar.
    ///
    /// # References
    /// This implementation is described in ["Faster Clustering Coefficient Using Vertex Covers"](https://ieeexplore.ieee.org/document/6693348).
    pub fn par_iter_clustering_coefficient_per_node(
        &self,
        verbose: Option<bool>,
    ) -> impl IndexedParallelIterator<Item = f64> + '_ {
        self.get_number_of_triangles_per_node(verbose)
            .into_par_iter()
            .zip(self.par_iter_node_degrees())
            .map(|(triangles_number, degree)| {
                if degree <= 1 {
                    0.0
                } else {
                    triangles_number as f64 / ((degree as EdgeT) * (degree as EdgeT - 1)) as f64
                }
            })
    }

    /// Returns clustering coefficients for all nodes in the graph.
    ///
    /// # Arguments
    /// * `low_centrality`: Option<usize> - The threshold over which to switch to parallel matryoshka. By default 50.
    /// * `verbose`: Option<bool> - Whether to show a loading bar.
    ///
    /// # References
    /// This implementation is described in ["Faster Clustering Coefficient Using Vertex Covers"](https://ieeexplore.ieee.org/document/6693348).
    pub fn get_clustering_coefficient_per_node(&self, verbose: Option<bool>) -> Vec<f64> {
        self.par_iter_clustering_coefficient_per_node(verbose)
            .collect()
    }

    /// Returns the graph clustering coefficient.
    ///
    /// # Arguments
    /// * `verbose`: Option<bool> - Whether to show a loading bar.
    ///
    /// # References
    /// This implementation is described in ["Faster Clustering Coefficient Using Vertex Covers"](https://ieeexplore.ieee.org/document/6693348).
    pub fn get_clustering_coefficient(&self, verbose: Option<bool>) -> f64 {
        self.par_iter_clustering_coefficient_per_node(verbose).sum()
    }

    /// Returns the graph average clustering coefficient.
    ///
    /// # Arguments
    /// * `verbose`: Option<bool> - Whether to show a loading bar.
    ///
    /// # References
    /// This implementation is described in ["Faster Clustering Coefficient Using Vertex Covers"](https://ieeexplore.ieee.org/document/6693348).
    pub fn get_average_clustering_coefficient(&self, verbose: Option<bool>) -> f64 {
        self.get_clustering_coefficient(verbose) / self.get_number_of_nodes() as f64
    }
}
