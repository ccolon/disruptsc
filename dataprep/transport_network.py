import re

import pandas as pd
from tqdm import tqdm


def get_merge_nodes(edges):
    end_points_connectivity = pd.concat([edges['end1'], edges['end2']]).value_counts()
    merge_points = end_points_connectivity[end_points_connectivity == 2].index.sort_values().to_list()
    return merge_points


def get_edges_from_endpoints(edges, endpoints: list):
    merged_df_end1 = pd.merge(pd.DataFrame({'endpoint': endpoints}),
                              edges[['id', 'end1']].rename(columns={'id': 'edge_id', 'end1': 'endpoint'}),
                              on='endpoint', how='left').dropna().astype(int)
    merged_df_end2 = pd.merge(pd.DataFrame({'endpoint': endpoints}),
                              edges[['id', 'end2']].rename(columns={'id': 'edge_id', 'end2': 'endpoint'}),
                              on='endpoint', how='left').dropna().astype(int)
    merged_df = pd.concat([merged_df_end1, merged_df_end2])
    return merged_df.groupby("endpoint")['edge_id'].apply(list).to_dict()


def check_degree2(node_id_to_edges_ids: dict):
    return (pd.Series(node_id_to_edges_ids).apply(len) == 2).all()


def merge_lines_with_tolerance(lines):
    """Snap lines to each other within a tolerance and then merge them."""
    snapped_lines = [snap(line, lines[0], TOLERANCE) for line in lines]  # Snap all lines to the first one
    return linemerge(snapped_lines)  # Merge the snapped lines


def merge_edges_attributes(gdf):
    new_data = {}
    new_data['geometry'] = linemerge(gdf.geometry.to_list())
    if new_data['geometry'].geom_type != "LineString":
        print(gdf.geometry)
        print(new_data['geometry'])
        raise ValueError(f"Merged geometry is: {new_data['geometry'].geom_type}")

    def string_to_list(s):
        return list(map(int, re.findall(r'\d+', s)))

    def merge_or_unique(column):
        unique_vals = gdf[column].dropna().unique()
        return unique_vals[0] if len(unique_vals) == 1 else ', '.join(unique_vals)

    # Aggregate columns based on given rules
    if 'km' in gdf.columns:
        new_data['km'] = gdf['km'].sum()
    if 'osmids' in gdf.columns:
        new_data['osmids'] = str(string_to_list(', '.join(map(str, gdf['osmids'].fillna('')))))
    if 'name' in gdf.columns:
        new_data['name'] = ', '.join(filter(None, gdf['name'].astype(str)))  # Ignore None values
    if 'capacity' in gdf.columns:
        new_data['capacity'] = gdf['capacity'].min()
    for col in ['end1', 'end2']:
        if col in gdf.columns:
            new_data[col] = None
    for col in ['special', 'class', 'surface', 'disruption']:
        if col in gdf.columns:
            new_data[col] = merge_or_unique(col)

    # Create a new row with the merged data
    return new_data


def update_gdf(gdf, new_data, old_ids, new_id):
    for col, value in new_data.items():
        gdf.at[new_id, col] = value
    gdf.loc[list(set(old_ids) - {new_id}), 'to_keep'] = False


def update_dict(my_dict, old_value, new_value):
    for key, value_list in my_dict.items():
        if old_value in value_list:
            my_dict[key] = [new_value if v == old_value else v for v in value_list]
    return my_dict


def remove_degree_2_nodes(edges):
    merge_nodes = get_merge_nodes(edges)
    print(f"Nb degree 2 nodes: {len(merge_nodes)}")
    merge_nodes_with_edges_ids = get_edges_from_endpoints(edges, merge_nodes)
    print(f"Check that they are actually 2 edges associated: {check_degree2(merge_nodes_with_edges_ids)}")

    edges['to_keep'] = True
    edges = edges.set_index('id')
    for merged_node in tqdm(list(merge_nodes_with_edges_ids.keys())):
        edge_ids = merge_nodes_with_edges_ids.pop(merged_node)
        print(edge_ids)
        merged_attributes = merge_edges_attributes(edges.loc[edge_ids])
        old_id = max(edge_ids)
        new_id = min(edge_ids)
        update_gdf(edges, merged_attributes, edge_ids, new_id)
        merge_nodes_with_edges_ids = update_dict(merge_nodes_with_edges_ids, old_id, new_id)

    print(
        f"Check that all resulting geometries are LineString: {edges['geometry'].apply(lambda geom: geom.geom_type == 'LineString').all()}")

    edges = edges[edges['to_keep']]
    edges = edges.drop(columns=['to_keep'])
    edges = edges.reset_index()
    return edges

# s = remove_degree_2_nodes(edges)