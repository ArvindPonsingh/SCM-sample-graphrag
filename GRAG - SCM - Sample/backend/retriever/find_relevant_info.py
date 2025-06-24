import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import sys
from collections import deque
import pickle 

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from backend.graph_builder.build_supply_chain_graph import load_data_and_build_graph

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def encode_node_attributes(graph, max_hops=2):
    
    node_embeddings = {}
    print(f"Starting node attribute encoding for {graph.number_of_nodes()} nodes with max_hops={max_hops}...")

    for i, (node, attributes) in enumerate(graph.nodes(data=True)):
        if i % 10 == 0 and i > 0: # Print progress
            print(f"  Processing node {i}/{graph.number_of_nodes()}...")

        text_to_encode_parts = []

        # Add node's own attributes
        node_type = attributes.get('type', 'node')
        text_to_encode_parts.append(f"This is a {node_type} with ID {node}.")
        for key, value in attributes.items():
            # Exclude specific keys that might be redundant or not useful for embedding
            if key not in ['type', 'Name', 'Description', 'Material', 'LeadTime', # Product
                           'Location', 'Reliability', # Supplier
                           'Source', 'Destination' # Shipment/Locations
                          ] and value is not None:
                str_value = str(value).strip()
                text_to_encode_parts.append(f"{key}: {str_value}.")

        # Add core identifying attributes explicitly to ensure they are always present
        if 'Name' in attributes and attributes['Name'] is not None:
            text_to_encode_parts.append(f"Name: {attributes['Name']}.")
        if 'Description' in attributes and attributes['Description'] is not None:
            text_to_encode_parts.append(f"Description: {attributes['Description']}.")
        if 'Location' in attributes and attributes['Location'] is not None:
            text_to_encode_parts.append(f"Location: {attributes['Location']}.")
        if 'Source' in attributes and attributes['Source'] is not None:
            text_to_encode_parts.append(f"Source: {attributes['Source']}.")
        if 'Destination' in attributes and attributes['Destination'] is not None:
            text_to_encode_parts.append(f"Destination: {attributes['Destination']}.")


        # Multi-hop neighbor information using BFS
        q = deque([(node, 0)]) # (current_node, current_hop)
        visited_nodes_for_embedding = {node} # Keep track of nodes already processed for embedding text
        processed_edges_for_embedding = set() # Prevent redundant edge descriptions

        while q:
            current_node_bfs, current_hop = q.popleft()

            if current_hop >= max_hops:
                continue

            # Iterate through neighbors (both outgoing and incoming for DiGraph)
            neighbors_this_hop = set(graph.neighbors(current_node_bfs)).union(set(graph.predecessors(current_node_bfs)))

            for neighbor_id in neighbors_this_hop:
                if neighbor_id == current_node_bfs: # Skip self-loops
                    continue

                # Add to queue for next hop if not visited before for BFS traversal
                if neighbor_id not in visited_nodes_for_embedding:
                    visited_nodes_for_embedding.add(neighbor_id)
                    q.append((neighbor_id, current_hop + 1))

                # --- Extract and format information about the neighbor and edge ---
                neighbor_type = graph.nodes[neighbor_id].get('type', 'entity')
                # Prioritize meaningful names for the neighbor, consolidating similar attributes
                neighbor_display_name = graph.nodes[neighbor_id].get('Name', '') or \
                                        graph.nodes[neighbor_id].get('ProductID', '') or \
                                        graph.nodes[neighbor_id].get('SupplierID', '') or \
                                        graph.nodes[neighbor_id].get('ShipmentID', '') or \
                                        graph.nodes[neighbor_id].get('Location', '') or \
                                        graph.nodes[neighbor_id].get('Source', '') or \
                                        graph.nodes[neighbor_id].get('Destination', '') or \
                                        str(neighbor_id) # Fallback to node ID

                # Describe the relationship from current_node_bfs to neighbor_id (outgoing)
                if graph.has_edge(current_node_bfs, neighbor_id):
                    edge_tuple = (current_node_bfs, neighbor_id)
                    if edge_tuple not in processed_edges_for_embedding:
                        edge_data = graph.get_edge_data(current_node_bfs, neighbor_id)
                        relation_type = edge_data.get('relation', 'connected to')
                        edge_attrs_list = []
                        for k, v in edge_data.items():
                            if k not in ['relation', 'ShipmentID', 'ProductID', 'SupplierID', 'SourceID', 'DestinationID'] and v is not None:
                                edge_attrs_list.append(f"{k}: {str(v).strip()}")
                        edge_attrs_str = ", ".join(edge_attrs_list)

                        text_to_encode_parts.append(f"From {current_node_bfs} it {relation_type} a {neighbor_type} identified as '{neighbor_display_name}'.")
                        if edge_attrs_str:
                            text_to_encode_parts.append(f"Connection details: {edge_attrs_str}.")
                        processed_edges_for_embedding.add(edge_tuple)

                # Describe the relationship from neighbor_id to current_node_bfs (incoming)
                if graph.has_edge(neighbor_id, current_node_bfs):
                    edge_tuple = (neighbor_id, current_node_bfs)
                    if edge_tuple not in processed_edges_for_embedding:
                        edge_data = graph.get_edge_data(neighbor_id, current_node_bfs)
                        relation_type = edge_data.get('relation', 'connected from')
                        edge_attrs_list = []
                        for k, v in edge_data.items():
                            if k not in ['relation', 'ShipmentID', 'ProductID', 'SupplierID', 'SourceID', 'DestinationID'] and v is not None:
                                edge_attrs_list.append(f"{k}: {str(v).strip()}")
                        edge_attrs_str = ", ".join(edge_attrs_list)

                        text_to_encode_parts.append(f"From a {neighbor_type} identified as '{neighbor_display_name}' (ID: {neighbor_id}) it is {relation_type} {current_node_bfs}.")
                        if edge_attrs_str:
                            text_to_encode_parts.append(f"Inverse connection details: {edge_attrs_str}.")
                        processed_edges_for_embedding.add(edge_tuple)

                # Add key attributes of the neighbor node itself, ensuring it's not the central node being processed
                if neighbor_id != node:
                    key_neighbor_attrs_list = []
                    for k_attr, v_attr in graph.nodes[neighbor_id].items():
                        # Exclude type and attributes already covered by neighbor_display_name or too generic
                        if k_attr not in ['type', 'Name', 'ProductID', 'SupplierID', 'ShipmentID',
                                          'Location', 'Source', 'Destination', 'Description',
                                          'Material', 'LeadTime', 'Reliability'] and v_attr is not None:
                            key_neighbor_attrs_list.append(f"{k_attr}:{str(v_attr).strip()}")
                    if key_neighbor_attrs_list:
                        text_to_encode_parts.append(f"The {neighbor_type} '{neighbor_display_name}' (ID: {neighbor_id}) has attributes: {', '.join(key_neighbor_attrs_list)}.")

        final_text = " ".join(text_to_encode_parts).strip()
        if final_text:
            node_embeddings[node] = model.encode(final_text)
        else:
            # Fallback for nodes with no attributes or connections, or very sparse ones
            node_embeddings[node] = np.zeros(model.get_sentence_embedding_dimension())

    print("Finished node attribute encoding.")
    return node_embeddings

def find_relevant_nodes(graph, query, node_embeddings, top_n=5, min_similarity_threshold=0.3):
    
    query_embedding = model.encode(query)
    similarity_scores = {}

    norm_query = np.linalg.norm(query_embedding)

    if norm_query == 0: # Handle empty query case
        print("Warning: Query embedding is zero, no meaningful similarity can be calculated.")
        return []

    for node, embedding in node_embeddings.items():
        norm_embedding = np.linalg.norm(embedding)

        if norm_embedding == 0:
            similarity = 0.0
        else:
            similarity = np.dot(query_embedding, embedding) / (norm_query * norm_embedding)

        similarity_scores[node] = similarity

    # Filter by similarity threshold first
    filtered_nodes = [(node, score) for node, score in similarity_scores.items() if score >= min_similarity_threshold]

    # Sort and take top_n
    sorted_nodes = sorted(filtered_nodes, key=lambda item: item[1], reverse=True)
    return [node for node, score in sorted_nodes[:top_n]]


def get_context_from_nodes(graph, relevant_nodes, max_neighbors_per_node=3):
    
    context_pieces = []

    if not relevant_nodes:
        return "No relevant information found in the graph for this query."

    context_pieces.append("--- Relevant Graph Information ---")

    for node in relevant_nodes:
        if not graph.has_node(node):
            continue # Skip if node somehow not in graph (e.g., deleted after embedding)

        # Format the attributes of the current relevant node
        node_type = graph.nodes[node].get('type', 'node')
        node_attrs_list = [f"{k}: {str(v).strip()}" for k, v in graph.nodes[node].items() if v is not None and k != 'type']
        attributes_str = f"Type: {node_type}, ID: {node}, Details: {', '.join(node_attrs_list)}"
        context_pieces.append(f"Node Info: {attributes_str}")

        processed_neighbors_for_context = set() # To avoid duplicate neighbor descriptions for the context

        # Outgoing neighbors (node -> neighbor)
        outgoing_neighbors_limited = list(graph.neighbors(node))[:max_neighbors_per_node]
        for neighbor in outgoing_neighbors_limited:
            if neighbor not in processed_neighbors_for_context and graph.has_node(neighbor):
                neighbor_type = graph.nodes[neighbor].get('type', 'entity')
                # Consolidate display names for neighbors as well
                neighbor_name = graph.nodes[neighbor].get('Name', None) or \
                                graph.nodes[neighbor].get('ProductID', None) or \
                                graph.nodes[neighbor].get('SupplierID', None) or \
                                graph.nodes[neighbor].get('ShipmentID', None) or \
                                graph.nodes[neighbor].get('Location', None) or \
                                graph.nodes[neighbor].get('Source', None) or \
                                graph.nodes[neighbor].get('Destination', None) or \
                                str(neighbor) # Fallback

                edge_data = graph.get_edge_data(node, neighbor)
                relation_parts = []
                if edge_data:
                    for k, v in edge_data.items():
                        if v is not None and k not in ['ShipmentID', 'ProductID', 'SupplierID', 'SourceID', 'DestinationID']:
                            relation_parts.append(f"{k}: {str(v).strip()}")
                relation_str = ", ".join(relation_parts) if relation_parts else "connected"

                context_pieces.append(
                    f"Relationship: {node_type} '{node}' --[{relation_str}]--> {neighbor_type} '{neighbor_name}' (ID: {neighbor})"
                )
                processed_neighbors_for_context.add(neighbor)

        # Incoming neighbors (neighbor -> node)
        incoming_neighbors_limited = list(graph.predecessors(node))[:max_neighbors_per_node]
        for neighbor in incoming_neighbors_limited:
            if neighbor not in processed_neighbors_for_context and graph.has_node(neighbor):
                neighbor_type = graph.nodes[neighbor].get('type', 'entity')
                neighbor_name = graph.nodes[neighbor].get('Name', None) or \
                                graph.nodes[neighbor].get('ProductID', None) or \
                                graph.nodes[neighbor].get('SupplierID', None) or \
                                graph.nodes[neighbor].get('ShipmentID', None) or \
                                graph.nodes[neighbor].get('Location', None) or \
                                graph.nodes[neighbor].get('Source', None) or \
                                graph.nodes[neighbor].get('Destination', None) or \
                                str(neighbor) # Fallback

                edge_data = graph.get_edge_data(neighbor, node) # Note: order is neighbor, node
                relation_parts = []
                if edge_data:
                    for k, v in edge_data.items():
                        if v is not None and k not in ['ShipmentID', 'ProductID', 'SupplierID', 'SourceID', 'DestinationID']:
                            relation_parts.append(f"{k}: {str(v).strip()}")
                relation_str = ", ".join(relation_parts) if relation_parts else "connected"

                context_pieces.append(
                    f"Relationship: {neighbor_type} '{neighbor_name}' (ID: {neighbor}) --[{relation_str}]--> {node_type} '{node}'"
                )
                processed_neighbors_for_context.add(neighbor)

    return "\n".join(context_pieces)

def find_relevant_info(graph, query, precomputed_node_embeddings=None, encode_max_hops=2):
    """
    Main function to find relevant information from the graph for a given query.
    Accepts precomputed_node_embeddings for performance optimization.
    encode_max_hops specifies the number of hops to consider during embedding if not precomputed.
    """
    if not graph or not graph.nodes():
        return "No graph data available or graph is empty."

    if precomputed_node_embeddings is not None:
        node_embeddings = precomputed_node_embeddings
    else:
        print(f"Warning: Node embeddings are being computed from scratch with max_hops={encode_max_hops}. For performance, consider precomputing and passing them.")
        node_embeddings = encode_node_attributes(graph, max_hops=encode_max_hops)

    # Lower the min_similarity_threshold slightly if you want to retrieve more nodes,
    # but be careful not to introduce too much noise.
    relevant_nodes = find_relevant_nodes(graph, query, node_embeddings, min_similarity_threshold=0.3)
    context = get_context_from_nodes(graph, relevant_nodes)
    return context


# --- Functions for Precomputation and Loading Embeddings ---
def precompute_and_save_embeddings(graph, output_path, max_hops=2):
    """
    Computes node embeddings for the entire graph and saves them to a file.
    """
    print(f"\n--- Starting Precomputation of Embeddings (max_hops={max_hops}) ---")
    embeddings = encode_node_attributes(graph, max_hops=max_hops)
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"--- Node embeddings saved to {output_path} ---")

def load_precomputed_embeddings(filepath):
    """
    Loads precomputed node embeddings from a file.
    """
    if os.path.exists(filepath):
        print(f"\n--- Loading precomputed node embeddings from {filepath} ---")
        try:
            with open(filepath, 'rb') as f:
                embeddings = pickle.load(f)
            print("--- Embeddings loaded successfully ---")
            return embeddings
        except Exception as e:
            print(f"Error loading embeddings from {filepath}: {e}. Will recompute.")
            return None
    else:
        print(f"\n--- No precomputed embeddings found at {filepath}. ---")
        return None

