import pandas as pd
from dfdiagnoser_ml.common import layer_key, get_feature_group

def save_shap_summary_to_csv(shap_results: dict, output_path: str):
    """
    Consolidates SHAP summary dataframes into a single CSV file.
    
    The CSV will have columns: pair_id, group, component, feature, 
    mean_abs_shap, mean_shap.
    """
    all_rows = []
    
    for pair_id, data in shap_results.items():
        if "summary_dataframes" not in data:
            continue
            
        summary_dfs = data["summary_dataframes"]
        
        # Loop through components (center, width, q_low, q_high)
        for component, group_dfs in summary_dfs.items():
            # Loop through groups (bad, good)
            for group, df in group_dfs.items():
                if df.empty:
                    continue
                
                # Add identifying columns
                df_to_append = df.reset_index()
                df_to_append['pair_id'] = pair_id
                df_to_append['group'] = group
                df_to_append['component'] = component
                
                all_rows.append(df_to_append)

    if not all_rows:
        print("No SHAP summary data to save.")
        return

    # Combine all data into a single DataFrame
    final_df = pd.concat(all_rows, ignore_index=True)
    
    # Reorder columns for clarity
    final_df = final_df[['pair_id', 'group', 'component', 'feature', 'mean_abs_shap', 'mean_shap']]
    
    # Save to CSV
    final_df.to_csv(output_path, index=False)
    print(f"Saved consolidated SHAP summary to: {output_path}")


def save_shap_layers_to_csv(shap_results: dict, output_path: str):
    """
    Aggregates feature-level SHAP summaries into layer-level summaries and saves to CSV.

    Output columns: pair_id, group, component, layer, mean_abs_shap, mean_shap
    where mean_abs_shap and mean_shap are sums across features within the layer.
    """
    all_rows = []

    for pair_id, data in shap_results.items():
        if "summary_dataframes" not in data:
            continue
        summary_dfs = data["summary_dataframes"]

        for component, group_dfs in summary_dfs.items():
            for group, df in group_dfs.items():
                if df.empty:
                    continue
                # df has index = feature
                df_feat = df.reset_index()  # columns: feature, mean_abs_shap, mean_shap
                if df_feat.empty:
                    continue
                # Map feature -> layer and aggregate sums
                df_feat['layer'] = df_feat['feature'].apply(layer_key)
                df_layer = df_feat.groupby('layer', as_index=False).agg({
                    'mean_abs_shap': 'sum',
                    'mean_shap': 'sum',
                })
                df_layer['pair_id'] = pair_id
                df_layer['group'] = group
                df_layer['component'] = component
                all_rows.append(df_layer[['pair_id', 'group', 'component', 'layer', 'mean_abs_shap', 'mean_shap']])

    if not all_rows:
        print("No SHAP layer data to save.")
        return

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df = final_df.sort_values(
        ['pair_id', 'group', 'component', 'mean_abs_shap'], 
        ascending=[True, True, True, False]
    )
    final_df.to_csv(output_path, index=False)
    print(f"Saved consolidated SHAP layers to: {output_path}")


def save_shap_feature_groups_to_csv(shap_results: dict, output_path: str):
    """
    Aggregates feature-level SHAP summaries into feature_group-level summaries and saves to CSV.

    Output columns: pair_id, group, component, feature_group, mean_abs_shap, mean_shap
    where mean_abs_shap and mean_shap are sums across features within the feature_group.
    """
    all_rows = []

    for pair_id, data in shap_results.items():
        if "summary_dataframes" not in data:
            continue
        summary_dfs = data["summary_dataframes"]

        for component, group_dfs in summary_dfs.items():
            for group, df in group_dfs.items():
                if df.empty:
                    continue
                # df has index = feature
                df_feat = df.reset_index()  # columns: feature, mean_abs_shap, mean_shap
                if df_feat.empty:
                    continue
                # Map feature -> feature_group and aggregate sums
                df_feat['feature_group'] = df_feat['feature'].apply(get_feature_group)
                df_fg = df_feat.groupby('feature_group', as_index=False).agg({
                    'mean_abs_shap': 'sum',
                    'mean_shap': 'sum',
                })
                df_fg['pair_id'] = pair_id
                df_fg['group'] = group
                df_fg['component'] = component
                all_rows.append(df_fg[['pair_id', 'group', 'component', 'feature_group', 'mean_abs_shap', 'mean_shap']])

    if not all_rows:
        print("No SHAP feature group data to save.")
        return

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df = final_df.sort_values(
        ['pair_id', 'group', 'component', 'mean_abs_shap'],
        ascending=[True, True, True, False]
    )
    final_df.to_csv(output_path, index=False)
    print(f"Saved consolidated SHAP feature groups to: {output_path}")


def save_shap_first_sample_to_csv(shap_results: dict, output_path: str):
    """
    Consolidates SHAP data for the first sample into a single CSV file.

    The CSV will have columns: pair_id, group, component, feature, 
    shap_value, abs_shap_value.
    """
    all_rows = []
    
    for pair_id, data in shap_results.items():
        if "first_sample" not in data:
            continue
            
        sample_dfs = data["first_sample"]
        
        # Loop through components (center, width, q_low, q_high)
        for component, group_dfs in sample_dfs.items():
            # Loop through groups (bad, good)
            for group, df in group_dfs.items():
                if df.empty:
                    continue
                
                # Add identifying columns
                df_to_append = df.reset_index()
                df_to_append['pair_id'] = pair_id
                df_to_append['group'] = group
                df_to_append['component'] = component
                
                all_rows.append(df_to_append)

    if not all_rows:
        print("No SHAP first sample data to save.")
        return

    # Combine all data into a single DataFrame
    final_df = pd.concat(all_rows, ignore_index=True)
    
    # Reorder columns for clarity
    final_df = final_df[['pair_id', 'group', 'component', 'feature', 'abs_shap_value', 'shap_value']]
    
    # Sort for clarity
    final_df = final_df.sort_values(
        ['pair_id', 'group', 'component', 'abs_shap_value'],
        ascending=[True, True, True, False]
    )

    # Save to CSV
    final_df.to_csv(output_path, index=False)
    print(f"Saved consolidated SHAP first sample data to: {output_path}")


def save_shap_io_bound_summary_to_csv(shap_results: dict, output_path: str):
    """
    Consolidates I/O-bound SHAP summary dataframes into a single CSV file.
    
    The CSV will have columns: pair_id, group, component, feature, 
    mean_abs_shap, mean_shap.
    """
    all_rows = []
    
    for pair_id, data in shap_results.items():
        if "io_bound_summary_dataframes" not in data:
            continue
            
        summary_dfs = data["io_bound_summary_dataframes"]
        
        # Loop through components (center, width, q_low, q_high)
        for component, group_dfs in summary_dfs.items():
            # Loop through groups (bad, good)
            for group, df in group_dfs.items():
                if df.empty:
                    continue
                
                # Add identifying columns
                df_to_append = df.reset_index()
                df_to_append['pair_id'] = pair_id
                df_to_append['group'] = group
                df_to_append['component'] = component
                
                all_rows.append(df_to_append)

    if not all_rows:
        print("No I/O-bound SHAP summary data to save.")
        return

    # Combine all data into a single DataFrame
    final_df = pd.concat(all_rows, ignore_index=True)
    
    # Reorder columns for clarity
    final_df = final_df[['pair_id', 'group', 'component', 'feature', 'mean_abs_shap', 'mean_shap']]
    
    # Sort for clarity
    final_df = final_df.sort_values(
        ['pair_id', 'group', 'component', 'mean_abs_shap'],
        ascending=[True, True, True, False]
    )

    # Save to CSV
    final_df.to_csv(output_path, index=False)
    print(f"Saved consolidated I/O-bound SHAP summary to: {output_path}")
