use super::*;

#[automatically_generated_function]
/// Returns whether the given node name respects the DrugBank nodes pattern.
///
/// # Arguments
/// * `node_name`: &str - Node name to check pattern with.
///
/// # Example
/// To validate a node you can use:
/// ```ignore
/// # use graph::*;
/// let this_library_node_name = "DRUGBANK:DB00999";
/// let not_this_library_node_name = "PizzaQuattroStagioni";
/// assert!(is_valid_drugbank_node_name(this_library_node_name));
/// assert!(!is_valid_drugbank_node_name(not_this_library_node_name));
/// ```
pub fn is_valid_drugbank_node_name(node_name: &str) -> bool {
    is_valid_node_name_from_seeds(
        node_name,
        Some(&["DRUGBANK"]),
        Some(16),
        Some(":"),
        Some("DB"),
        Some(7),
        Some(5),
    )
    .is_ok()
}

#[automatically_generated_function]
/// Returns URL from given DrugBank node name.
///
/// # Arguments
/// * `node_name`: &str - Node name to check pattern with.
///
/// # Safety
/// This method assumes that the provided node name is a DrugBank node name and
/// may cause a panic if the aforementioned assumption is not true.
///
pub(crate) unsafe fn format_drugbank_url_from_node_name(node_name: &str) -> String {
    format_url_from_node_name(
        "http://identifiers.org/drugbank/{node_name}",
        node_name,
        Some(":"),
    )
}
