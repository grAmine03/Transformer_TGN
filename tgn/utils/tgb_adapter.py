import numpy as np
import pandas as pd
from data_processing import Data
from tgb.linkproppred.dataset import LinkPropPredDataset

def get_tgb_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False, use_validation=False):
    """
    Adapter for TGB datasets like "tgbl-comment" or "tgbl-review"
    """
    

    dataset = LinkPropPredDataset(name=dataset_name, root="datasets", preprocess=True)
    
    # Extract data correctly
    df = dataset.full_data
    sources = df['sources'].to_numpy()
    destinations = df['destinations'].to_numpy()
    timestamps = df['timestamps'].to_numpy()
    edge_idxs = df['edge_idxs'].to_numpy()
    labels = df['edge_label'].to_numpy() # All 1s in TGB data normally, negatives are generated later
    
    node_features = dataset.node_feat
    if node_features is None:
        # Default empty node features if absent
        max_node = max(sources.max(), destinations.max())
        node_features = np.zeros((max_node + 1, 100)) # Default 100 dim
    
    edge_features = dataset.edge_feat
    if edge_features is None:
        edge_features = np.zeros((len(sources) + 1, 100))
    elif len(edge_features) == len(sources):
        # Pad with one row for the "0" edge index
        edge_features = np.vstack([np.zeros(edge_features.shape[1]), edge_features])
        
    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    
    # Masks
    val_time, test_time = list(np.quantile(timestamps, [0.70, 0.85]))
    
    import random
    random.seed(2020)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # Compute nodes which appear at test time
    test_node_set = set(sources[timestamps > val_time]).union(
        set(destinations[timestamps > val_time]))
    # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
    # their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

    # Mask saying for each source and destination whether they are new test nodes
    new_test_source_mask = np.isin(sources, list(new_test_node_set))
    new_test_destination_mask = np.isin(destinations, list(new_test_node_set))
    
    # Mask which is true for edges with both destination and source not being new test nodes (because
    # we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
    
    # For train we keep edges happening before the validation time which do not involve any new node
    # used for inductiveness
    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
    
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time
    
    if different_new_nodes_between_val_and_test:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

        edge_contains_new_val_node_mask = np.array(
          [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array(
          [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
    else:
        edge_contains_new_node_mask = np.array(
          [(a in new_test_node_set or b in new_test_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)
        
    # validation and test with all edges
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask],
                             edge_idxs[new_node_val_mask], labels[new_node_val_mask])

    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                              labels[new_node_test_mask])

    return full_data, node_features, edge_features, train_data, val_data, test_data, \
           new_node_val_data, new_node_test_data
