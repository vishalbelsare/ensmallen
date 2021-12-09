use super::*;

#[automatically_generated_function]
/// Returns whether the given node name respects the C. elegans phenotype nodes pattern.
///
/// # Arguments
/// * `node_name`: &str - Node name to check pattern with.
///
/// # Example
/// To validate a node you can use:
/// ```ignore
/// # use graph::*;
/// let this_library_node_name = "WBPhenotype:0010002";
/// let not_this_library_node_name = "PizzaQuattroStagioni";
/// assert!(is_valid_c_elegans_phenotype_node_name(this_library_node_name));
/// assert!(!is_valid_c_elegans_phenotype_node_name(not_this_library_node_name));
/// ```
pub fn is_valid_c_elegans_phenotype_node_name(node_name: &str) -> bool {
    is_valid_node_name_from_seeds(
        node_name,
        Some(&["WBPHENOTYPE"]),
        Some(19),
        Some(":"),
        None,
        Some(7),
        Some(7),
    )
    .is_ok()
}

#[automatically_generated_function]
/// Returns URL from given C. elegans phenotype node name.
///
/// # Arguments
/// * `node_name`: &str - Node name to check pattern with.
///
/// # Safety
/// This method assumes that the provided node name is a C. elegans phenotype node name and
/// may cause a panic if the aforementioned assumption is not true.
///
pub(crate) unsafe fn format_c_elegans_phenotype_url_from_node_name(node_name: &str) -> String {
    format_url_from_node_name(
        "http://purl.obolibrary.org/obo/WBPHENOTYPE_{node_name}",
        node_name,
        Some(":"),
    )
}
