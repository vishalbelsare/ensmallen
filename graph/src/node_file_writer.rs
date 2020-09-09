use super::*;

/// Structure that saves the parameters specific to writing and reading a nodes csv file.
///
/// # Attributes
/// * parameters: CSVFile - The common parameters for readin and writing a csv.
/// * nodes_column: String - The name of the nodes names column. This parameter is mutually exclusive with nodes_column_number.
/// * nodes_column_number: usize - The rank of the column with the nodes names. This parameter is mutually exclusive with nodes_column.
/// * node_types_column: String - The name of the nodes type column. This parameter is mutually exclusive with node_types_column_number.
/// * node_types_column_number: usize - The rank of the column with the nodes types. This parameter is mutually exclusive with node_types_column.
pub struct NodeFileWriter {
    pub(crate) parameters: CSVFileWriter,
    pub(crate) nodes_column: String,
    pub(crate) node_types_column: String,
    pub(crate) nodes_column_number: usize,
    pub(crate) node_types_column_number: usize,
}

impl NodeFileWriter {
    /// Return new NodeFileWriter object.
    ///
    /// # Arguments
    ///
    /// * parameters: CSVFileParameters - Path where to store/load the file.
    ///
    pub fn new(parameters: CSVFileWriter) -> NodeFileWriter {
        NodeFileWriter {
            parameters,
            nodes_column: "id".to_string(),
            nodes_column_number: 0,
            node_types_column: "category".to_string(),
            node_types_column_number: 1,
        }
    }

    /// Set the column of the nodes.
    ///
    /// # Arguments
    ///
    /// * nodes_column: Option<String> - The nodes column to use for the file.
    ///
    pub fn set_nodes_column(
        mut self,
        nodes_column: Option<String>,
    ) -> Result<NodeFileWriter, String> {
        if let Some(v) = nodes_column {
            self.nodes_column = v;
        }
        Ok(self)
    }

    /// Set the column of the nodes.
    ///
    /// # Arguments
    ///
    /// * node_types_column: Option<String> - The node types column to use for the file.
    ///
    pub fn set_node_types_column(
        mut self,
        nodes_type_column: Option<String>,
    ) -> Result<NodeFileWriter, String> {
        if let Some(v) = nodes_type_column {
            self.node_types_column = v;
        }
        Ok(self)
    }

    /// Set the column_number of the nodes.
    ///
    /// # Arguments
    ///
    /// * nodes_column_number: Option<usize> - The nodes column_number to use for the file.
    ///
    pub fn set_nodes_column_number(mut self, nodes_column_number: Option<usize>) -> NodeFileWriter {
        if let Some(v) = nodes_column_number {
            self.nodes_column_number = v;
        }
        self
    }

    /// Set the column_number of the nodes.
    ///
    /// # Arguments
    ///
    /// * node_types_column_number: Option<usize> - The node types column_number to use for the file.
    ///
    pub fn set_node_types_column_number(
        mut self,
        node_types_column_number: Option<usize>,
    ) -> NodeFileWriter {
        if let Some(v) = node_types_column_number {
            self.node_types_column_number = v;
        }
        self
    }

    /// Read node file and returns graph builder data structures.
    ///  
    pub(crate) fn write_node_file(
        &self,
        nodes_reverse_mapping:&Vec<String>,
        node_types:&Option<Vec<NodeTypeT>>,
        node_types_reverse_mapping:&Option<Vec<String>>
    ) -> Result<(), String> {
        // build the header
        let mut header = vec![(
            self.nodes_column,
            self.nodes_column_number
        )];
        let number_of_columns = 1 + if node_types.is_some() {
            header.push((self.node_types_column, self.node_types_column_number));
            max!(self.nodes_column_number, self.node_types_column_number)
        } else {
            self.nodes_column_number
        };

        match node_types {
            Some(nts)=> {
                if let Some(ntrm) = node_types_reverse_mapping {
                    self.parameters.write_lines(
                        nodes_reverse_mapping.len() as u64,
                        compose_lines(number_of_columns, header),
                        
                        nodes_reverse_mapping.iter()
                        .zip(
                            nts.iter().map(
                                |node_type| ntrm[*node_type as usize]
                            )
                        ).map(
                            |(node_name, node_type)| 
                            compose_lines(
                                number_of_columns,
                                vec![
                                    (*node_name, self.nodes_column_number),
                                    (node_type, self.node_types_column_number)
                                ]
                            )
                        )
                    )
                } else {
                    unreachable!()
                }
            },
            None => self.parameters.write_lines(
                    nodes_reverse_mapping.len() as u64,
                    compose_lines(number_of_columns, header),
                    nodes_reverse_mapping.iter().map(
                        |node_name| 
                        compose_lines(
                            number_of_columns,
                            vec![(*node_name, self.nodes_column_number)]
                        )
                    )
                )
            }
    }
}
