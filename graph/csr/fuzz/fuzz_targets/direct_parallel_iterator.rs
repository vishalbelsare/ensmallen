//! Fuzzing harness to test whether the cardinality estimation works as expected.
#![no_main]

use arbitrary::Arbitrary;
use csr::*;
use libfuzzer_sys::fuzz_target;
use rayon::prelude::*;

#[derive(Arbitrary, Debug)]
struct FuzzCase {
    // We use u16 so it cannot create
    // an object that is too big
    edges: Vec<(u16, u16)>,
    number_of_nodes: u16,
}

fuzz_target!(|data: FuzzCase| {
    let mut edges = data.edges.clone();

    // sort edges
    edges.sort_unstable();

    // Identify the number of nodes, and adjust accordingly
    let number_of_nodes = (edges
        .iter()
        .copied()
        .map(|(src, dst)| src.max(dst))
        .max()
        .unwrap_or(0) as usize
        + 1)
    .max(data.number_of_nodes as usize);

    // Build the CSR
    let csrb = ConcurrentCSRBuilder::new(edges.len() as u64, number_of_nodes as u32);

    edges
        .iter()
        .copied()
        .enumerate()
        .for_each(|(i, (src, dst))| {
            csrb.set(i as u64, src as u32, dst as u32);
        });

    let csr = csrb.build();

    // Check that the number of nodes is correct
    assert_eq!(csr.get_number_of_nodes(), number_of_nodes as u32);

    // Check that the number of edges is correct
    assert_eq!(csr.get_number_of_directed_edges(), edges.len() as u64);

    // Check that the edges are correct
    assert_eq!(
        csr.par_iter_directed_edge_node_ids()
            .map(|(_, src, dst)| (src, dst))
            .collect::<Vec<_>>(),
        edges
            .iter()
            .copied()
            .map(|(src, dst)| (src as u32, dst as u32))
            .collect::<Vec<_>>()
    );
});
