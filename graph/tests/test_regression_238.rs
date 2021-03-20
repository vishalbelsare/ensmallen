extern crate graph;

use graph::{Graph, EdgeFileReader};

#[test]
/// This is a regression test that has been automatically generated
/// by the fuzzer harness.
/// The test originally caused a panic in the file edge_type_vocabulary.rs",
/// specifically (at the time) line 49 and column 16.
///
/// The report of the graph, if available, is:
/// The directed graph \u{0}\u{0}kk has 2 nodes and 2 unweighted edges with
/// a single edge type: kkkk, of which 1 are self-loops.
/// The graph is extremely dense as it has a density of 0.50000
/// and is connected, as it has a single component.
/// The graph median node degree is 2, the mean node degree is 1.00,
/// and the node degree mode is 2. The top 2 most central nodes are kkkk
/// (degree 2) and kkkkkk\u{1} (degree 0).
/// The hash of the graph is 12d94077c09fcf3f.")
///
fn test_regression_238() -> Result<(), String> {
    let edges_reader = EdgeFileReader::new("tests/data/regression/238.edges")?
        .set_rows_to_skip(Some(0))
        .set_header(Some(false))
        .set_separator(Some(","))?
        .set_verbose(Some(false))
        .set_ignore_duplicates(Some(false))
        .set_skip_self_loops(Some(false))
        .set_numeric_edge_type_ids(Some(false))
        .set_numeric_node_ids(Some(false))
        .set_skip_weights_if_unavailable(Some(false))
        .set_skip_edge_types_if_unavailable(Some(false));

    let nodes_reader = None;

    let mut graph = Graph::from_unsorted_csv(
        edges_reader,
        nodes_reader,
        true, // Directed
        true, // Directed edge list
        "  kk" // Name of the graph
    )?;

    let _ = graph::test_utilities::default_test_suite(&mut graph, false);
    Ok(())
}
