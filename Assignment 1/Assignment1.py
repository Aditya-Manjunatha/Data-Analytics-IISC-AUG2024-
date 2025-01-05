# Importing the required libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist, squareform
########################################################################################################################################################################################################################
#                                                                                Importing the Data sets

def import_wiki_vote_data(file_path):
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue  
            if line.strip():
                i, j = map(int, line.split())  
                edges.append(tuple(sorted((i, j))))

    # Step 4: Convert to numpy array
    edges_array = np.array(edges)
    
    # Step 5: Remove duplicate edges
    unique_edges = np.unique(edges_array, axis=0)
    
    return unique_edges

def import_lastfm_asia_data(file_path):
    # Step 1: Load the CSV file, skipping the header
    df = pd.read_csv(file_path, header=0)
    
    # Step 2: Convert the DataFrame to a list of tuples
    edges = []
    for index, row in df.iterrows():
        i, j = int(row[0]), int(row[1])
        edges.append(tuple(sorted((i, j))))

    # Step 3: Convert to numpy array
    edges_array = np.array(edges)
    
    # Step 4: Remove duplicate edges
    unique_edges = np.unique(edges_array, axis=0)
    
    
    return unique_edges       

########################################################################################################################################################################################################################                                                                                                    
#                                                                                Girvan Newman Algorithm

def compute_modularity(G, communities):
    m = len(G.edges())  # Total number of edges
    modularity = 0

    for com in communities:
        subgraph = G.subgraph(com)
        ki = {node: len(list(G.neighbors(node))) for node in com} # Degree of each node in the community
        
        # Calculate internal edges
        internal_edges = 0
        for u, v in subgraph.edges():
            internal_edges += 1
        
        # Calculate the modularity contribution for this community
        for u in com:
            for v in com:
                if G.has_edge(u, v):
                    ki_u = ki[u]
                    ki_v = ki[v]
                    # Contribution to modularity from edge (u, v)
                    modularity += 1 - (ki_u * ki_v) / (2 * m)
    
    # Normalize modularity
    return modularity / (2 * m)

def calculate_btw_and_communities(G):

    edge_btw_dict = {}
    #com_res = set()
    vertices = list(G.nodes())
    for v in vertices:
        visited = []
        src = [v]
        node_score = {node: [0, 1] for node in G.nodes()}
        node_score[v][0] += 1
        edge_path = []
    # Now for each vertex, we start a BFS traversal
        while True:
            visited.extend(src)
            next_src = set()
            cur_level_edge = {}

            for current_node in src:
                neighbours = list(G.neighbors(current_node))
                for next_node in neighbours:
                    if next_node not in visited:
                        next_src.add(next_node)
                        node_score[next_node][0] += node_score[current_node][0]
                        if next_node not in cur_level_edge:
                            cur_level_edge[next_node] = []
                        cur_level_edge[next_node].append(current_node)
            if len(next_src) == 0:
                #if G.degree(v) == 0:
                    #com_res.add(tuple([v]))
                #else:
                    #com_res.add(tuple(sorted(visited)))
                break
            else:
                edge_path.append(cur_level_edge)
                src = next_src
        for indx in range(len(edge_path)-1, -1, -1):
            level_nodes = edge_path[indx]
            for current_node in level_nodes:
                edges = level_nodes[current_node]
                for next_node in edges:
                    btw_val = node_score[current_node][1] * node_score[next_node][0] / node_score[current_node][0]
                    edge = tuple(sorted([current_node, next_node]))
                    if edge not in edge_btw_dict:
                        edge_btw_dict[edge] = 0
                    edge_btw_dict[edge] += btw_val
                    node_score[next_node][1] += btw_val
    for e in edge_btw_dict:
        edge_btw_dict[e] /= 2
    
    return edge_btw_dict

def count_communities(graph_partition):
    # Flatten the graph_partition array to a 1D array and find unique values
    unique_communities = np.unique(graph_partition)
    
    # Count the number of unique community IDs
    num_communities = len(unique_communities)
    
    return num_communities

def partition_to_communities(graph_partition):
    unique_communities = np.unique(graph_partition)
    communities = []
    for community_id in unique_communities:
        community = set(np.where(graph_partition == community_id)[0])
        communities.append(community)
    return communities

def remove_highest_btw_edge(G, edge_btw_dict):
    # Find the edge with the highest betweenness centrality
    edge_to_remove = max(edge_btw_dict, key=edge_btw_dict.get)
        
        # Remove the edge from the graph
    G.remove_edge(*edge_to_remove)
        
        
    print(f"Removed edge: {edge_to_remove}")
    return G

def Girvan_Newman_one_level(nodes_connectivity_list):
    G= nx.Graph()
    G.add_edges_from(nodes_connectivity_list)
    # Initialize the graph_partition array with -1
    graph_partition = np.full((G.number_of_nodes(), 1), -1)
    
    # Get the initial number of connected components
    initial_connected_components = nx.number_connected_components(G)
    
    while nx.number_connected_components(G) == initial_connected_components:
        # Calculate the edge betweenness for all edges
        edge_betweenness = calculate_btw_and_communities(G)
        
        G = remove_highest_btw_edge(G, edge_betweenness)
        
    
    # Now, find the connected components and assign community IDs
    connected_components = list(nx.connected_components(G))
    for component in connected_components:
        smallest_node_id = min(component)
        for node in component:
            graph_partition[node][0] = smallest_node_id
    
    return graph_partition

def Girvan_Newman(nodes_connectivity_list):
    
    G = nx.Graph()
    G.add_edges_from(nodes_connectivity_list)
    A = np.full((G.number_of_nodes(), 1), -1)
    stop_criteria = 0
    old_partition = np.arange(G.number_of_nodes()).reshape(G.number_of_nodes(), 1)
    
    while True:
        print(f"Stop criteria: {stop_criteria}")
        
        old_communities = partition_to_communities(old_partition)
        old_modularity = compute_modularity(G, old_communities)
        #print(f"Old modularity: {old_modularity}")
        #print(f"Old partition: {old_partition}")
        #print(f"Old communities: {old_communities}")

        new_nodes_connectivity_list = list(G.edges())
        # Perform one level of the Girwan-Newman algorithm
        new_graph_partition = Girvan_Newman_one_level(new_nodes_connectivity_list)  
        new_communities = partition_to_communities(new_graph_partition)  # Is a list of sets
        #print(f"New Communities: {new_communities}")
        #print(f"New partition: {new_graph_partition}")
        # Append the new partition to A
        A = np.hstack((A, new_graph_partition))
        
        # Calculate current modularity
        new_modularity = compute_modularity(G, new_communities)
        
        #print(f"New modularity: {new_modularity}")
        
        # Check if the modularity has increased
        if new_modularity > old_modularity:
            stop_criteria += 1
            print(f"modularity change  = {new_modularity - old_modularity}")
            old_partition = new_graph_partition
            old_modularity = new_modularity
            
        else:
            # Revert to the old partition and remove the last column from A
            new_graph_partition = old_partition
            new_communities = old_communities
            A = A[:, :-1]
            print(f"modularity change = {new_modularity - old_modularity}")
            print(f"Modularity decreased so reverted to old partition {new_graph_partition} and old communities {new_communities}")
            break
    
    return A

def visualise_dendogram(community_mat):

    A = community_mat
    print("Shape of community matrix A:", A.shape)

    # Flatten the matrix for distance computation
    # Each row should be a single vector representing the community assignments of a node
    flattened_A = A

    # Compute the pairwise Euclidean distance between nodes
    dist_matrix = pdist(flattened_A, metric='euclidean')
    dist_matrix = squareform(dist_matrix)
    print("Shape of distance matrix:", dist_matrix.shape)

    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(dist_matrix, method='ward')

    # Create a dendrogram plot
    plt.figure(figsize=(12, 8))
    dendro = sch.dendrogram(linkage_matrix, labels=np.arange(flattened_A.shape[0]), orientation='top', leaf_rotation=90, leaf_font_size=10)
    
    plt.title('Dendrogram')
    plt.xlabel('Node')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.show()


########################################################################################################################################################################################################################
#                                                                                Louvain Algorithm

def count_no_communities(communities):
    unique_values = set(communities.values())  
    return len(unique_values)

def calculate_modularity_change(G, communities, node, neighbor_comm, m):
    old_community = communities[node]
    old_community_members = [n for n in G.nodes() if communities[n] == old_community]
    neighbor_comm_members = [n for n in G.nodes() if communities[n] == neighbor_comm]
    k_i = G.degree(node)  # Degree of the current node
    K_i_X = sum(1 for neighbour in G.neighbors(node) if neighbour in old_community_members)  # Number of edges between the current node and its current community
    k_i_Y = sum(1 for neighbour in G.neighbors(node) if neighbour in neighbor_comm_members)
    S_X = sum(G.degree(i) for i in old_community_members) - k_i
    S_Y = sum(G.degree(i) for i in neighbor_comm_members)
    
    delta_Q = (k_i_Y - K_i_X) / m + k_i * (S_X - S_Y) / (2 * m **2)
    return delta_Q

def louvain_one_iter(nodes_connectivity_list):

    G= nx.Graph()
    #nodes_connectivity_list is a tuple of edges 
    G.add_edges_from(nodes_connectivity_list)
    # Step 1: Initialize each node to its own community
    communities = {node: node for node in G.nodes()}  # Each node starts in its own community
    A = np.full((G.number_of_nodes(), 1), -1)
    m = G.number_of_edges()  # Total number of edges in the graph
    print(f"Total number of edges in the graph: {m}")

    # Step 2: Iteratively evaluate each node's neighbors for modularity gain
    improvement = True
    while improvement:
        improvement = False
        for i in G.nodes():
            current_comm = communities[i]
            neighbors = list(G.neighbors(i))

            #print(f"Current Node considered is {i}, in community {current_comm}:") 

            best_comm = current_comm
            best_delta_Q = 0
            
            for j in neighbors:
                comm_j = communities[j]
                
                if comm_j == current_comm:
                    continue

                # Calculate delta Q
                delta_Q = calculate_modularity_change(G, communities, i, comm_j, m)
                
                #print(f"Considering moving node {i} from community {current_comm} to {comm_j} gives delta_Q={delta_Q}")

                if delta_Q > best_delta_Q:
                    best_delta_Q = delta_Q
                    best_comm = comm_j
            
            # If there's an improvement, move the node to the best community
            if best_comm != current_comm:
                communities[i] = best_comm
                improvement = True

            # Update community IDs in the old community
                old_community_nodes = [node for node in G.nodes() if communities[node] == current_comm]
                if old_community_nodes:
                    new_id_old_comm = min(old_community_nodes)
                    for node in old_community_nodes:
                        communities[node] = new_id_old_comm

                # Update community IDs in the new community
                new_community_nodes = [node for node in G.nodes() if communities[node] == best_comm]
                if new_community_nodes:
                    new_id_new_comm = min(new_community_nodes)
                    for node in new_community_nodes:
                        communities[node] = new_id_new_comm
                #print(f"Updated community assignments: {communities}")
                graph_partition_array = np.array([communities[node] for node in G.nodes()]).reshape(-1, 1)
                A = np.hstack((A, graph_partition_array))
    #print(f"The community matrix is \n {A[:, 1:]}")
    no_of_communities = count_no_communities(communities)

    return graph_partition_array

########################################################################################################################################################################################################################
#                                                                                Calling the functions

if __name__ == "__main__":

    nodes_connectivity_list_wiki = import_wiki_vote_data("../data/Wiki-Vote.txt")

    graph_partition_wiki  = Girvan_Newman_one_level(nodes_connectivity_list_wiki)

    community_mat_wiki = Girvan_Newman(nodes_connectivity_list_wiki)

    visualise_dendogram(community_mat_wiki)

    graph_partition_louvain_wiki = louvain_one_iter(nodes_connectivity_list_wiki)


    nodes_connectivity_list_lastfm = import_lastfm_asia_data("../data/lastfm_asia_edges.csv")

    graph_partition_lastfm = Girvan_Newman_one_level(nodes_connectivity_list_lastfm)

    community_mat_lastfm = Girvan_Newman(nodes_connectivity_list_lastfm)
    
    visualise_dendogram(community_mat_lastfm)
    
    graph_partition_louvain_lastfm = louvain_one_iter(nodes_connectivity_list_lastfm)

########################################################################################################################################################################################################################



