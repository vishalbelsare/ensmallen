use super::*;
use rayon::prelude::*;

/// # Queries
/// The naming convection we follow is `get_X_by_Y`.
impl Graph {

    /// Return the src, dst, edge type and weight of a given edge id
    pub fn get_edge_quadruple(
        &self,
        edge_id: EdgeT,
    ) -> (NodeT, NodeT, Option<EdgeTypeT>, Option<WeightT>) {
        let (src, dst, edge_type) = self.get_edge_triple(edge_id);
        (src, dst, edge_type, self.get_unchecked_weight_by_edge_id(edge_id))
    }

    /// Return the src, dst, edge type of a given edge id
    pub fn get_edge_triple(&self, edge_id: EdgeT) -> (NodeT, NodeT, Option<EdgeTypeT>) {
        let (src, dst) = self.get_node_ids_from_edge_id(edge_id);
        (src, dst, self.get_unchecked_edge_type_by_edge_id(edge_id))
    }

    /// Return vector with top k central node Ids.
    ///
    /// # Arguments
    ///
    /// * k: NodeT - Number of central nodes to extract.
    pub fn get_top_k_central_nodes_ids(&self, k: NodeT) -> Vec<NodeT> {
        let mut nodes_degrees: Vec<(NodeT, NodeT)> = (0..self.get_nodes_number())
            .map(|node_id| (self.get_node_degree_by_node_id(node_id).unwrap(), node_id))
            .collect();
        nodes_degrees.par_sort_unstable();
        nodes_degrees.reverse();
        nodes_degrees[0..k as usize]
            .iter()
            .map(|(_, node_id)| *node_id)
            .collect()
    }


    /// Return vector with top k central node names.
    ///
    /// # Arguments
    ///
    /// * k: NodeT - Number of central nodes to extract.
    pub fn get_top_k_central_node_names(&self, k: NodeT) -> Vec<String> {
        self.get_top_k_central_nodes_ids(k)
            .into_iter()
            .map(|node_id| self.get_node_name_by_node_id(node_id).unwrap())
            .collect()
    }

    /// Returns node type of given node.
    ///
    /// # Arguments
    ///
    /// * `node_id`: NodeT - node whose node type is to be returned.
    ///
    /// # Examples
    /// ```rust
    /// # let graph = graph::test_utilities::load_ppi(true, true, true, true, false, false).unwrap();
    /// println!("The node type id of node {} is {:?}", 0, graph.get_node_type_id_by_node_id(0));
    /// ```
    ///
    pub fn get_node_type_id_by_node_id(&self, node_id: NodeT) -> Result<Option<Vec<NodeTypeT>>, String> {
        if let Some(nt) = &self.node_types {
            return if node_id <= nt.ids.len() as NodeT {
                Ok(nt.ids[node_id as usize].clone())
            } else {
                Err(format!(
                    "The node_index {} is too big for the node_types vector which has len {}",
                    node_id,
                    nt.ids.len()
                ))
            };
        }

        Err(String::from(
            "Node types are not defined for current graph instance.",
        ))
    }

    /// Returns edge type of given edge.
    ///
    /// # Arguments
    ///
    /// * edge_id: EdgeT - edge whose edge type is to be returned.
    ///
    /// # Examples
    /// ```rust
    /// # let graph = graph::test_utilities::load_ppi(true, true, true, true, false, false).unwrap();
    /// println!("The edge type id of edge {} is {:?}", 0, graph.get_edge_type_id_by_edge_id(0));
    /// ```
    pub fn get_edge_type_id_by_edge_id(&self, edge_id: EdgeT) -> Result<Option<EdgeTypeT>, String> {
        if let Some(et) = &self.edge_types {
            return if edge_id <= et.ids.len() as EdgeT {
                Ok(self.get_unchecked_edge_type_by_edge_id(edge_id))
            } else {
                Err(format!(
                    "The edge_index {} is too big for the edge_types vector which has len {}",
                    edge_id,
                    et.ids.len()
                ))
            };
        }
        Err(String::from(
            "Edge types are not defined for current graph instance.",
        ))
    }

    /// Returns option with the node type of the given node id.
    /// TODO: MOST LIKELY THIS SHOULD BE CHANGED!!!
    pub fn get_node_type_name_by_node_id(&self, node_id: NodeT) -> Result<Option<Vec<String>>, String> {
        match &self.node_types.is_some() {
            true => Ok(match self.get_unchecked_node_type_id_by_node_id(node_id) {
                Some(node_type_id) => Some(self.get_node_type_names_by_node_type_ids(node_type_id)?),
                None => None,
            }),
            false => Err("Node types not available for the current graph instance.".to_string()),
        }
    }

    /// Returns option with the edge type of the given edge id.
    /// TODO: complete docstring and add example!
    /// TODO: THIS SHOULD RETURN A RESULT!
    pub fn get_edge_type_name_by_edge_id(&self, edge_id: EdgeT) -> Result<Option<String>, String> {
        self.get_edge_type_id_by_edge_id(edge_id)?
            .map_or(
                Ok(None),
                |x| Ok(Some(self.get_edge_type_name_by_edge_type_id(x)?))
        )
    }

    /// Return edge type name of given edge type.
    ///
    /// # Arguments
    /// * edge_type_id: EdgeTypeT - Id of the edge type.
    pub fn get_edge_type_name_by_edge_type_id(&self, edge_type_id: EdgeTypeT) -> Result<String, String> {
        self.edge_types
            .as_ref()
            .map_or(
                Err("Edge types not available for the current graph instance.".to_string()),
                |ets| ets.translate(edge_type_id))
        
    }

    /// Returns weight of the given edge id.
    ///
    /// # Arguments
    /// * `edge_id`: EdgeT - The edge ID whose weight is to be returned.
    ///
    /// # Examples
    /// To get the weight of a given `edge_id` you can run:
    /// ```rust
    /// # let weighted_graph = graph::test_utilities::load_ppi(true, true, true, true, false, false).unwrap();
    /// # let unweighted_graph = graph::test_utilities::load_ppi(true, true, false, true, false, false).unwrap();
    /// let edge_id = 0;
    /// let unexistent_edge_id = 123456789;
    /// assert!(weighted_graph.get_weight_by_edge_id(edge_id).is_ok());
    /// assert!(weighted_graph.get_weight_by_edge_id(unexistent_edge_id).is_err());
    /// assert!(unweighted_graph.get_weight_by_edge_id(edge_id).is_err());
    /// ```
    pub fn get_weight_by_edge_id(&self, edge_id: EdgeT) -> Result<WeightT, String> {
        self.weights.as_ref().map_or(
            Err("The current graph instance does not have weights!".to_string()),
            |weights| weights.get(edge_id as usize).map_or(
                Err(format!(
                    "The given edge_id {} is higher than the number of available directed edges {}.",
                    edge_id,
                    self.get_directed_edges_number()
                )),
                |value| Ok(*value)
            )
        )
    }

    /// Returns weight of the given node ids.
    ///
    /// # Arguments
    /// * `src`: NodeT - The node ID of the source node.
    /// * `dst`: NodeT - The node ID of the destination node.
    ///
    /// # Examples
    /// To get the weight of a given `src` and `dst` you can run:
    /// ```rust
    /// # let weighted_graph = graph::test_utilities::load_ppi(false, true, true, true, false, false).unwrap();
    /// let src = 0;
    /// let dst = 1;
    /// assert!(weighted_graph.get_weight_by_node_ids(src, dst).is_ok());
    /// ```
    pub fn get_weight_by_node_ids(&self, src: NodeT, dst: NodeT) -> Result<WeightT, String> {
        self.get_weight_by_edge_id(self.get_edge_id_by_node_ids(src, dst)?)
    }

    /// Returns weight of the given node ids and edge type.
    ///
    /// # Arguments
    /// * `src`: NodeT - The node ID of the source node.
    /// * `dst`: NodeT - The node ID of the destination node.
    /// * `edge_type`: Option<EdgeTypeT> - The edge type ID of the edge.
    ///
    /// # Examples
    /// To get the weight of a given `src` and `dst` and `edge_type` you can run:
    /// ```rust
    /// # let weighted_graph = graph::test_utilities::load_ppi(false, true, true, true, false, false).unwrap();
    /// let src = 0;
    /// let dst = 1;
    /// let edge_type = Some(0);
    /// assert!(weighted_graph.get_weight_with_type_by_node_ids(src, dst, edge_type).is_ok());
    /// ```
    pub fn get_weight_with_type_by_node_ids(&self, src: NodeT, dst: NodeT, edge_type: Option<EdgeTypeT>) -> Result<WeightT, String> {
        self.get_weight_by_edge_id(self.get_edge_id_with_type_by_node_ids(src, dst, edge_type)?)
    }

    /// Returns weight of the given node names and edge type.
    ///
    /// # Arguments
    /// * `src`: &str - The node name of the source node.
    /// * `dst`: &str - The node name of the destination node.
    /// * `edge_type`: Option<&String> - The edge type name of the edge.
    ///
    /// # Examples
    /// To get the weight of a given `src` and `dst` and `edge_type` you can run:
    /// ```rust
    /// # let weighted_graph = graph::test_utilities::load_ppi(false, true, true, true, false, false).unwrap();
    /// let src = "ENSP00000000233";
    /// let dst = "ENSP00000432568";
    /// let edge_type = Some("red".to_string());
    /// assert!(weighted_graph.get_weight_with_type_by_node_names(src, dst, edge_type.as_ref()).is_ok());
    /// ```
    pub fn get_weight_with_type_by_node_names(&self, src: &str, dst: &str, edge_type: Option<&String>) -> Result<WeightT, String> {
        self.get_weight_by_edge_id(self.get_edge_id_with_type_by_node_names(src, dst, edge_type)?)
    }

    /// Returns weight of the given node names.
    ///
    /// # Arguments
    /// * `src_name`: &str - The node name of the source node.
    /// * `dst_name`: &str - The node name of the destination node.
    ///
    /// # Examples
    /// To get the weight of a given `src_name` and `dst_name` you can run:
    /// ```rust
    /// # let weighted_graph = graph::test_utilities::load_ppi(false, true, true, true, false, false).unwrap();
    /// let src_name = "ENSP00000000233";
    /// let dst_name = "ENSP00000432568";
    /// assert!(weighted_graph.get_weight_by_node_names(src_name, dst_name).is_ok());
    /// ```
    pub fn get_weight_by_node_names(&self, src_name: &str, dst_name: &str) -> Result<WeightT, String> {
        self.get_weight_by_edge_id(self.get_edge_id_by_node_names(src_name, dst_name)?)
    }

    /// Returns result with the node name.
    pub fn get_node_name_by_node_id(&self, node_id: NodeT) -> Result<String, String> {
        match node_id < self.get_nodes_number() {
            true => Ok(self.nodes.unchecked_translate(node_id)),
            false => Err(format!(
                "Given node_id {} is greater than number of nodes in the graph ({}).",
                node_id,
                self.get_nodes_number()
            )),
        }
    }

    /// Returns result with the node id.
    pub fn get_node_id_by_node_name(&self, node_name: &str) -> Result<NodeT, String> {
        match self.nodes.get(node_name) {
            Some(node_id) => Ok(*node_id),
            None => Err(format!(
                "Given node name {} is not available in current graph.",
                node_name
            )),
        }
    }

    /// Return node type ID for the given node name if available.
    ///
    /// # Arguments
    /// 
    /// * `node_name`: &str - Name of the node.
    ///
    /// # Examples
    /// To get the node type ID for a given node name you can run:
    /// ```rust
    /// # let graph = graph::test_utilities::load_ppi(true, true, true, true, false, false).unwrap();
    /// let node_name = "ENSP00000000233";
    /// println!("The node type ID of node {} is {:?}.", node_name, graph.get_node_type_id_by_node_name(node_name).unwrap());
    /// ```
    pub fn get_node_type_id_by_node_name(&self, node_name: &str) -> Result<Option<Vec<NodeTypeT>>, String> {
        self.get_node_type_id_by_node_id(self.get_node_id_by_node_name(node_name)?)
    }

    /// Return node type name for the given node name if available.
    ///
    /// # Arguments
    /// 
    /// * `node_name`: &str - Name of the node.
    ///
    /// # Examples
    /// To get the node type name for a given node name you can run:
    /// ```rust
    /// # let graph = graph::test_utilities::load_ppi(true, true, true, true, false, false).unwrap();
    /// let node_name = "ENSP00000000233";
    /// println!("The node type of node {} is {:?}", node_name, graph.get_node_type_name_by_node_name(node_name).unwrap());
    /// ```
    pub fn get_node_type_name_by_node_name(&self, node_name: &str) -> Result<Option<Vec<String>>, String> {
        self.get_node_type_name_by_node_id(self.get_node_id_by_node_name(node_name)?)
    }

    /// TODO: add doc
    pub fn get_edge_count_by_edge_type_id(&self, edge_type: Option<EdgeTypeT>) -> Result<EdgeT, String> {
        if !self.has_edge_types() {
            return Err("Current graph does not have edge types!".to_owned());
        }
        if let Some(et) = &edge_type{
            if self.get_edge_types_number() <= *et {
                return Err(format!(
                    "Given edge type ID {} is bigger than number of edge types in the graph {}.",
                    self.get_edge_types_number(),
                    et
                ));
            }
        }
        Ok(self.get_unchecked_edge_count_by_edge_type_id(edge_type))
    }
    
    /// TODO: add doc
    pub fn get_edge_type_id_by_edge_type_name(&self, edge_type_name: Option<&str>) -> Result<Option<EdgeTypeT>, String> {
        match (&self.edge_types, edge_type_name) {
            (None, _) => Err("Current graph does not have edge types.".to_owned()),
            (Some(_), None) => Ok(None),
            (Some(ets), Some(etn)) => {
                match ets.get(etn) {
                    Some(edge_type_id) => Ok(Some(*edge_type_id)),
                    None => Err(format!(
                        "Given edge type name {} is not available in current graph.",
                        etn
                    )),
                }
            }
        }
    }

    /// TODO: add doc
    pub fn get_edge_count_by_edge_type_name(&self, edge_type: Option<&str>) -> Result<EdgeT, String> {
        self.get_edge_count_by_edge_type_id(self.get_edge_type_id_by_edge_type_name(edge_type)?)
    }

    /// TODO: add doc
    pub fn get_node_type_id_by_node_type_name(&self, node_type_name: &str) -> Result<NodeTypeT, String> {
        if let Some(ets) = &self.node_types {
            return match ets.get(node_type_name) {
                Some(node_type_id) => Ok(*node_type_id),
                None => Err(format!(
                    "Given node type name {} is not available in current graph.",
                    node_type_name
                )),
            };
        }
        Err("Current graph does not have node types.".to_owned())
    }

    /// TODO: add doc
    pub fn get_node_count_by_node_type_id(&self, node_type: Option<NodeTypeT>) -> Result<NodeT, String> {
        if !self.has_node_types() {
            return Err("Current graph does not have node types!".to_owned());
        }
        if node_type.map_or(false, |nt| self.get_node_types_number() <= nt) {
            return Err(format!(
                "Given node type ID {:?} is bigger than number of node types in the graph {}.",
                node_type,
                self.get_node_types_number()
            ));
        }
        Ok(self.get_unchecked_node_count_by_node_type_id(node_type))
    }

    /// TODO: add docstring
    pub fn get_node_count_by_node_type_name(&self, node_type_name: Option<&str>) -> Result<NodeT, String> {
        self.get_node_count_by_node_type_id(node_type_name.map_or(
            Ok::<_, String>(None), 
            |ntn| Ok(Some(self.get_node_type_id_by_node_type_name(ntn)?))
        )?)
    }

    /// TODO!: add unchecked version of this method!
    /// TODO: add docstring and example!
    pub fn get_destination_node_id_by_edge_id(&self, edge_id: EdgeT) -> Result<NodeT, String> {
        if edge_id >= self.get_directed_edges_number(){
            return Err(format!(
                "The edge ID {} is higher than the number of available directed edges {}.",
                edge_id,
                self.get_directed_edges_number()
            ));
        }
        Ok(match &self.destinations {
            Some(destinations) => destinations[edge_id as usize],
            None => self.get_node_ids_from_edge_id(edge_id).1,
        })
    }

    /// Return vector of destinations for the given source node ID.
    ///
    /// # Arguments
    ///
    /// * `node_id`: NodeT - Node ID whose neighbours are to be retrieved.
    ///
    /// # Example
    /// To retrieve the neighbours of a given node `src` you can use:
    /// 
    /// ```rust
    /// # let graph = graph::test_utilities::load_ppi(true, true, true, true, false, false).unwrap();
    /// # let node_id = 0;
    /// println!("The neighbours of the node {} are {:?}.", node_id, graph.get_node_neighbours_by_node_id(node_id).unwrap());
    /// let unavailable_node = 2349765432;
    /// assert!(graph.get_node_neighbours_by_node_id(unavailable_node).is_err());
    /// ```
    pub fn get_node_neighbours_by_node_id(&self, node_id: NodeT) -> Result<Vec<NodeT>, String> {
        if node_id >= self.get_nodes_number(){
            return Err(format!(
                "The node ID {} is higher than the number of available nodes {}.",
                node_id,
                self.get_nodes_number()
            ));
        }
        Ok(self.iter_unchecked_edge_ids_by_source_node_id(node_id)
        .map(move |edge_id| self.get_destination_node_id_by_edge_id(edge_id).unwrap()).collect())
    }

    /// Return vector of destinations for the given source node name.
    ///
    /// # Arguments
    ///
    /// * `node_name`: &str - Node ID whose neighbours are to be retrieved.
    ///
    /// # Example
    /// To retrieve the neighbours of a given node `src` you can use:
    /// 
    /// ```rust
    /// # let graph = graph::test_utilities::load_ppi(true, true, true, true, false, false).unwrap();
    /// let node_id = 0;
    /// println!("The neighbours of the node {} are {:?}.", node_id, graph.get_node_neighbours_by_node_id(node_id).unwrap());
    /// ```
    pub fn get_node_neighbours_by_node_name(&self, node_name: &str) -> Result<Vec<NodeT>, String> {
        self.get_node_neighbours_by_node_id(self.get_node_id_by_node_name(node_name)?)
    }

    /// Return vector of destination names for the given source node name.
    ///
    /// # Arguments
    ///
    /// * `node_id`: NodeT - Node ID whose neighbours are to be retrieved.
    ///
    /// # Example
    /// To retrieve the neighbours of a given node `src` you can use:
    /// 
    /// ```rust
    /// # let graph = graph::test_utilities::load_ppi(true, true, true, true, false, false).unwrap();
    /// let node_name = "ENSP00000000233";
    /// println!("The neighbours of the node {} are {:?}.", node_name, graph.get_node_neighbours_name_by_node_name(node_name).unwrap());
    /// ```
    pub fn get_node_neighbours_name_by_node_name(&self, node_name: &str) -> Result<Vec<String>, String> {
        Ok(self.iter_node_neighbours(self.get_node_id_by_node_name(node_name)?).collect())
    }

    /// Return edge ID without any checks for given tuple of nodes and edge type.
    /// 
    /// This method will return an error if the graph does not contain the
    /// requested edge with edge type.
    ///
    /// # Arguments
    /// `src`: NodeT - Source node of the edge. 
    /// `dst`: NodeT - Destination node of the edge.
    /// `edge_type`: Option<EdgeTypeT> - Edge Type of the edge.
    ///
    pub fn get_edge_id_with_type_by_node_ids(
        &self,
        src: NodeT,
        dst: NodeT,
        edge_type: Option<EdgeTypeT>,
    ) -> Result<EdgeT, String> {
        let edge_id = self.edge_types.as_ref().map_or_else(|| self.get_edge_id_by_node_ids(src, dst).ok(), |ets| self.get_edge_ids_by_node_ids(src, dst).and_then(|mut edge_ids| {
            edge_ids.find(|edge_id| ets.ids[*edge_id as usize] == edge_type)
        }));
        // TODO: change using a map_err!
        match edge_id{
            Some(e) => Ok(e),
            None => Err(
                format!(
                    concat!(
                        "The current graph instance does not contain the required edge composed of ",
                        "source node ID {}, destination node ID {} and edge ID {:?}."
                    ),
                    src, dst, edge_type
                )
            )
        }
    }

    // TODO: add docstring and example!
    pub fn get_edge_id_by_node_names(
        &self,
        src_name: &str,
        dst_name: &str
    ) -> Result<EdgeT, String> {
        // TODO REFACTOR CODE to be cleaner!
        let edge_id = if let (Some(src), Some(dst)) = (self.nodes.get(src_name), self.nodes.get(dst_name)) {
            self.get_edge_id_by_node_ids(*src, *dst).ok()
        } else {
            None
        };
        match edge_id {
            Some(e) => Ok(e),
            None => Err(
                format!(
                    concat!(
                        "The current graph instance does not contain the required edge composed of ",
                        "source node name {} and destination node name {}."
                    ),
                    src_name, dst_name
                )
            )
        }
    }

    // TODO: add docstring and example!
    pub fn get_edge_id_with_type_by_node_names(
        &self,
        src_name: &str,
        dst_name: &str,
        edge_type_name: Option<&String>,
    ) -> Result<EdgeT, String> {
        if let (Some(src), Some(dst)) = (self.nodes.get(src_name), self.nodes.get(dst_name)) {
            self.get_edge_id_with_type_by_node_ids(*src, *dst, self.get_edge_type_id_by_edge_type_name(edge_type_name.map(|x| x.as_str()))?)
        } else {
            Err(
                format!(
                    concat!(
                        "The current graph instance does not contain the required edge composed of ",
                        "source node name {}, destination node name {} and edge name {:?}."
                    ),
                    src_name, dst_name, edge_type_name
                )
            )
        }
    }

    /// Return translated edge types from string to internal edge ID.
    ///
    /// # Arguments
    ///
    /// * `edge_types`: Vec<String> - Vector of edge types to be converted.
    pub fn get_edge_type_ids_by_edge_type_names(
        &self,
        edge_types: Vec<Option<String>>,
    ) -> Result<Vec<Option<EdgeTypeT>>, String> {
        match &self.edge_types {
                None => Err(String::from("Current graph does not have edge types.")),
                Some(ets) => {
                    edge_types
                    .iter()
                    .map(|edge_type_name|
                        match edge_type_name {
                            None=> Ok(None),
                            Some(et) => {
                                match ets.get(et) {
                                    Some(edge_type_id) => Ok(Some(*edge_type_id)),
                                    None => Err(format!(
                                        "The edge type {} does not exist in current graph. The available edge types are {}.",
                                        et,
                                        ets.keys().join(", ")
                                    ))
                                }
                            }
                        }
                    )
                .collect::<Result<Vec<Option<EdgeTypeT>>, String>>()
            }
        }
    }

    /// Return translated node types from string to internal node ID.
    ///
    /// # Arguments
    ///
    /// * `node_types`: Vec<String> - Vector of node types to be converted.
    pub fn get_node_type_ids_by_node_type_names(&self, node_types: Vec<Option<String>>) -> Result<Vec<Option<NodeTypeT>>, String> {
        match &self.node_types {
            None => Err(String::from("Current graph does not have node types.")),
            Some(nts) => {
                node_types
                .iter()
                .map(|node_type_name| 
                    match node_type_name {
                        None => Ok(None),
                        Some(nt) => {
                            match nts.get(nt) {
                                Some(node_type_id) => Ok(Some(*node_type_id)),
                                None => Err(format!(
                                    "The node type {} does not exist in current graph. The available node types are {}.",
                                    nt,
                                    nts.keys().join(", ")
                                )),
                            }
                        }
                    })
                .collect::<Result<Vec<Option<NodeTypeT>>, String>>()
            }
        }
    }

    /// Return range of outbound edges IDs for all the edges bewteen the given
    /// source and destination nodes.
    /// This operation is meaningfull only in a multigraph.
    ///
    /// # Arguments
    ///
    /// * src: NodeT - Source node.
    /// * dst: NodeT - Destination node.
    ///
    pub(crate) fn get_minmax_edge_ids_by_node_ids(
        &self,
        src: NodeT,
        dst: NodeT,
    ) -> Option<(EdgeT, EdgeT)> {
        self.get_edge_id_by_node_ids(src, dst).ok().map(
            |min_edge|
            (min_edge, self.get_unchecked_edge_id_from_tuple(src, dst + 1))
        )
    }

    /// Return range of outbound edges IDs which have as source the given Node.
    ///
    /// # Arguments
    ///
    /// * src: NodeT - Node for which we need to compute the outbounds range.
    ///
    pub(crate) fn get_minmax_edge_ids_by_source_node_id(&self, src: NodeT) -> (EdgeT, EdgeT) {
        match &self.outbounds {
            Some(outbounds) => {
                let min_edge_id = if src == 0 {
                    0
                } else {
                    outbounds[src as usize - 1]
                };
                (min_edge_id, outbounds[src as usize])
            }
            None => {
                let min_edge_id: EdgeT = self.get_unchecked_edge_id_from_tuple(src, 0);
                (
                    min_edge_id,
                    match &self.cached_destinations {
                        Some(cds) => match cds.get(&src) {
                            Some(destinations) => destinations.len() as EdgeT + min_edge_id,
                            None => self.get_unchecked_edge_id_from_tuple(src + 1, 0),
                        },
                        None => self.get_unchecked_edge_id_from_tuple(src + 1, 0),
                    },
                )
            }
        }
    }

    /// Returns option of range of multigraph minimum and maximum edge ids with same source and destination nodes and different edge type.
    ///
    /// # Arguments
    /// 
    /// * `src` - Source node of the edge.
    /// 
    pub fn get_edge_ids_by_node_ids(&self, src: NodeT, dst: NodeT) -> Option<impl Iterator<Item = EdgeT>> {
        self.get_minmax_edge_ids_by_node_ids(src, dst)
            .map(|(min_edge_id, max_edge_id)| min_edge_id..max_edge_id)
    }

    /// Return node type name of given node type.
    ///
    /// There is no need for a unchecked version since we will have to map
    /// on the note_types anyway.
    /// 
    /// # Arguments
    /// * node_type_id: Vec<NodeTypeT> - Id of the node type.
    pub fn get_node_type_name_by_node_type_id(&self, node_type_id: NodeTypeT) -> Result<String, String> {
        self.node_types
            .as_ref()
            .map_or(
                Err("Node types not available for the current graph instance.".to_string()),
                |nts| nts.translate(node_type_id)
            )
    }

    /// Return node type name of given node type.
    ///
    /// # Arguments
    /// * node_type_ids: Vec<NodeTypeT> - Id of the node type.
    pub fn get_node_type_names_by_node_type_ids(
        &self,
        node_type_ids: Vec<NodeTypeT>,
    ) -> Result<Vec<String>, String> {
        self.node_types.as_ref().map_or(
            Err("Node types not available for the current graph instance.".to_string()), 
            |nts| {
                nts.translate_vector(node_type_ids)
        })
    }

    /// Returns the number of outbound neighbours of given node.
    ///
    /// This is implemented as proposed by [S. Vigna here](http://vigna.di.unimi.it/ftp/papers/Broadword.pdf).
    ///
    /// # Arguments
    ///
    /// * `node_id` - Integer ID of the node.
    ///
    pub fn get_node_degree_by_node_id(&self, node_id: NodeT) -> Result<NodeT, String> {
        if node_id >= self.get_nodes_number(){
            return Err(format!(
                "The node ID {} is higher than the number of available nodes {}.",
                node_id,
                self.get_nodes_number()
            ));
        }
        let (min_edge_id, max_edge_id) = self.get_minmax_edge_ids_by_source_node_id(node_id);
        Ok((max_edge_id - min_edge_id) as NodeT)
    }
}
