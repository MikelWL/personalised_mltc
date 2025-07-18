"""
Cross-population dataset handling for CPRD + SAIL analysis.

This module provides functionality to:
1. Identify and manage combined dataset options
2. Load and merge corresponding CPRD and SAIL datasets
3. Handle cross-population analysis workflows
"""

import pandas as pd
from typing import Tuple, Optional, Dict, Any


def get_combined_dataset_options():
    """
    Generate combined dataset options for the dropdown.
    
    Returns:
        list: Combined dataset identifiers in format 'COMBINED_[demographic]'
    """
    # Define the demographic groups that have both CPRD and SAIL versions
    demographic_groups = [
        ('Females_45to64', 'Females 45 to 64 years'),
        ('Females_65plus', 'Females 65 years and over'),
        ('Females_below45', 'Females below 45 years'),
        ('Males_45to64', 'Males 45 to 64 years'),
        ('Males_65plus', 'Males 65 years and over'),
        ('Males_below45', 'Males below 45 years')
    ]
    
    combined_options = []
    for demo_code, demo_label in demographic_groups:
        combined_id = f"COMBINED_{demo_code}"
        combined_options.append(combined_id)
    
    return combined_options


def get_readable_combined_filename(combined_id):
    """
    Convert combined dataset ID to human-readable format.
    
    Args:
        combined_id (str): Combined dataset identifier (e.g., 'COMBINED_Females_45to64')
        
    Returns:
        str: Human-readable format (e.g., 'CPRD + SAIL Females 45 to 64 years')
    """
    # Mapping from demographic codes to labels
    demo_mapping = {
        'Females_45to64': 'Females 45 to 64 years',
        'Females_65plus': 'Females 65 years and over',
        'Females_below45': 'Females below 45 years',
        'Males_45to64': 'Males 45 to 64 years',
        'Males_65plus': 'Males 65 years and over',
        'Males_below45': 'Males below 45 years'
    }
    
    if not combined_id.startswith('COMBINED_'):
        return combined_id
    
    demo_code = combined_id.replace('COMBINED_', '')
    demo_label = demo_mapping.get(demo_code, demo_code)
    
    return f"CPRD + SAIL {demo_label}"


def is_combined_dataset(dataset_id):
    """
    Check if a dataset ID represents a combined dataset.
    
    Args:
        dataset_id (str): Dataset identifier
        
    Returns:
        bool: True if combined dataset, False otherwise
    """
    return dataset_id.startswith('COMBINED_')


def get_component_files(combined_id):
    """
    Get the CPRD and SAIL component files for a combined dataset.
    
    Args:
        combined_id (str): Combined dataset identifier
        
    Returns:
        tuple: (cprd_file, sail_file) or (None, None) if invalid
    """
    if not is_combined_dataset(combined_id):
        return None, None
    
    demo_code = combined_id.replace('COMBINED_', '')
    
    # CPRD file (CPRD_ prefix)
    cprd_file = f"CPRD_{demo_code}.csv"
    
    # SAIL file (SAIL_ prefix, with standard case)
    if demo_code.startswith('Females_') or demo_code.startswith('Males_'):
        # Convert Females_45to64 -> SAIL_Females_45to64
        sail_file = f"SAIL_{demo_code}.csv"
    else:
        sail_file = f"SAIL_{demo_code}.csv"
    
    return cprd_file, sail_file


def load_combined_dataset(combined_id, load_function):
    """
    Load and combine CPRD and SAIL datasets for analysis.
    
    Args:
        combined_id (str): Combined dataset identifier
        load_function (callable): Function to load individual datasets (e.g., load_and_process_data)
        
    Returns:
        tuple: (combined_data, total_patients, gender, age_group) or (None, None, None, None) if error
    """
    try:
        cprd_file, sail_file = get_component_files(combined_id)
        if not cprd_file or not sail_file:
            raise ValueError(f"Invalid combined dataset ID: {combined_id}")
        
        # Load both datasets
        cprd_data, cprd_patients, cprd_gender, cprd_age = load_function(cprd_file)
        sail_data, sail_patients, sail_gender, sail_age = load_function(sail_file)
        
        if cprd_data is None or sail_data is None:
            return None, None, None, None
        
        # Naive concatenation for now (will be enhanced later)
        combined_data = combine_datasets_naive(cprd_data, sail_data)
        
        # Combine population metadata
        total_patients = cprd_patients + sail_patients
        gender = cprd_gender  # Should be the same for both
        age_group = cprd_age  # Should be the same for both
        
        return combined_data, total_patients, gender, age_group
        
    except Exception as e:
        print(f"Error loading combined dataset {combined_id}: {str(e)}")
        return None, None, None, None


def combine_datasets_naive(cprd_data, sail_data):
    """
    Perform naive concatenation of CPRD and SAIL datasets.
    This version no longer overwrites the TotalPatientsInGroup column,
    preserving the original patient counts for each subset.
    
    Args:
        cprd_data (pd.DataFrame): CPRD dataset
        sail_data (pd.DataFrame): SAIL dataset
        
    Returns:
        pd.DataFrame: Concatenated dataset
    """
    # Add population source identifier
    cprd_data_labeled = cprd_data.copy()
    sail_data_labeled = sail_data.copy()
    
    cprd_data_labeled['PopulationSource'] = 'CPRD'
    sail_data_labeled['PopulationSource'] = 'SAIL'
    
    # Concatenate datasets, preserving original TotalPatientsInGroup values
    combined_data = pd.concat([cprd_data_labeled, sail_data_labeled], ignore_index=True)
    
    return combined_data


def is_feature_available_for_combined(feature_name):
    """
    Check if a feature is available for combined datasets.
    
    Args:
        feature_name (str): Name of the feature/tab
        
    Returns:
        bool: True if available, False otherwise
    """
    # For now, only Condition Combinations is implemented
    available_features = ['Condition Combinations']
    return feature_name in available_features


def get_unavailable_feature_message(feature_name):
    """
    Get the message to display for unavailable features.
    
    Args:
        feature_name (str): Name of the unavailable feature
        
    Returns:
        str: Message explaining feature availability
    """
    return f"""
    ## {feature_name} - Not Yet Available for Combined Datasets
    
    This feature has not been implemented for combined CPRD + SAIL datasets yet.
    
    **Currently Available for Combined Datasets:**
    - Condition Combinations Analysis
    
    **To use this feature:**
    1. Select a single dataset (CPRD or SAIL) from the dropdown
    2. Or check back later as we continue implementing cross-population features
    
    **Why this limitation?**
    Combined dataset analysis requires specialized statistical methods to properly 
    account for population differences and ensure valid comparisons.
    """


def is_combined_dataset_data(data):
    """
    Check if a loaded dataset is a combined dataset based on its structure.
    
    Args:
        data (pd.DataFrame): Loaded dataset
        
    Returns:
        bool: True if combined dataset, False otherwise
    """
    return 'PopulationSource' in data.columns


def analyze_condition_combinations_cross_population(data, min_percentage, min_frequency, original_analysis_func, shared_only=False):
    """
    Analyze condition combinations across populations with comparative metrics.
    This version correctly handles the total patient counts for each subset.
    
    Args:
        data (pd.DataFrame): Combined dataset with PopulationSource column
        min_percentage (float): Minimum prevalence threshold
        min_frequency (int): Minimum pair frequency threshold
        original_analysis_func (callable): Original single-population analysis function
        shared_only (bool): If True, only show combinations present in both populations.
        
    Returns:
        pd.DataFrame: Cross-population analysis results with CPRD, SAIL, and Combined columns
    """
    # Split data by population source
    cprd_data = data[data['PopulationSource'] == 'CPRD'].copy()
    sail_data = data[data['PopulationSource'] == 'SAIL'].copy()
    
    # Run original analysis function on each subset (they have correct, separate totals)
    cprd_results = original_analysis_func(cprd_data, min_percentage, min_frequency)
    sail_results = original_analysis_func(sail_data, min_percentage, min_frequency)
    
    # Add rank columns based on prevalence before merging
    if not cprd_results.empty:
        cprd_results['CPRD Rank'] = cprd_results['Prevalence of the combination (%)'].rank(method='min', ascending=False).astype(int)
    if not sail_results.empty:
        sail_results['SAIL Rank'] = sail_results['Prevalence of the combination (%)'].rank(method='min', ascending=False).astype(int)

    # For the combined analysis, we must first create a temporary dataframe
    # with the correct combined total patient count.
    combined_analysis_data = data.copy()
    cprd_total = cprd_data['TotalPatientsInGroup'].iloc[0] if not cprd_data.empty else 0
    sail_total = sail_data['TotalPatientsInGroup'].iloc[0] if not sail_data.empty else 0
    combined_analysis_data['TotalPatientsInGroup'] = cprd_total + sail_total
    
    combined_results = original_analysis_func(combined_analysis_data, min_percentage, min_frequency)
    
    # Merge results into comparative format
    return merge_cross_population_results(cprd_results, sail_results, combined_results, shared_only)


def merge_cross_population_results(cprd_results, sail_results, combined_results, shared_only=False):
    """
    Merge results from CPRD, SAIL, and combined analyses into comparative format.
    This version includes rank comparison columns and uses 'Both' instead of 'Combined'.
    Optionally filters to show only combinations present in both populations and re-ranks.

    Args:
        cprd_results (pd.DataFrame): CPRD analysis results with rank
        sail_results (pd.DataFrame): SAIL analysis results with rank
        combined_results (pd.DataFrame): Combined analysis results
        shared_only (bool): If True, only show combinations present in both populations.
        
    Returns:
        pd.DataFrame: Merged results with cross-population comparison columns
    """
    # Create dictionaries for easy lookup
    cprd_dict = {row['Combination']: row for _, row in cprd_results.iterrows()}
    sail_dict = {row['Combination']: row for _, row in sail_results.iterrows()}
    combined_dict = {row['Combination']: row for _, row in combined_results.iterrows()}
    
    # Get all unique combinations that appear in the combined results
    all_combinations = combined_dict.keys()
    
    # Build merged results
    merged_results = []
    
    for combination in all_combinations:
        cprd_row = cprd_dict.get(combination)
        sail_row = sail_dict.get(combination)
        combined_row = combined_dict.get(combination)
        
        # This check is redundant if all_combinations is from combined_dict, but safe to keep
        if combined_row is None:
            continue

        # Apply shared_only filter
        if shared_only:
            if cprd_row is None or sail_row is None:
                continue # Skip if not present in both

        # Calculate ranks and difference
        cprd_rank = cprd_row['CPRD Rank'] if cprd_row is not None else None
        sail_rank = sail_row['SAIL Rank'] if sail_row is not None else None
        
        rank_diff = None
        if cprd_rank is not None and sail_rank is not None:
            rank_diff = cprd_rank - sail_rank

        # Build merged row
        merged_row = {
            'Combination': combination,
            'Num': combined_row['NumConditions'],
            'CPRD Rank': cprd_rank,
            'SAIL Rank': sail_rank,
            'Rank Diff': rank_diff,
            
            # Minimum Pair Frequency columns
            'CPRD MPF': cprd_row['Minimum Pair Frequency'] if cprd_row is not None else 0,
            'SAIL MPF': sail_row['Minimum Pair Frequency'] if sail_row is not None else 0,
            'Both MPF': combined_row['Minimum Pair Frequency'],
            
            # Prevalence percentage columns
            'CPRD %': cprd_row['Prevalence of the combination (%)'] if cprd_row is not None else 0.0,
            'SAIL %': sail_row['Prevalence of the combination (%)'] if sail_row is not None else 0.0,
            'Both %': combined_row['Prevalence of the combination (%)'],
            
            # Odds ratio columns
            'CPRD OR': cprd_row['Total odds ratio'] if cprd_row is not None else 0.0,
            'SAIL OR': sail_row['Total odds ratio'] if sail_row is not None else 0.0,
            'Both OR': combined_row['Total odds ratio']
        }
        
        merged_results.append(merged_row)
    
    # Create DataFrame and sort by combined prevalence
    merged_df = pd.DataFrame(merged_results)
    
    # Handle potential empty DataFrame
    if not merged_df.empty:
        # Re-calculate ranks if shared_only filter was applied
        if shared_only:
            if not merged_df.empty:
                merged_df['CPRD Rank'] = merged_df['CPRD %'].rank(method='min', ascending=False).astype(int)
                merged_df['SAIL Rank'] = merged_df['SAIL %'].rank(method='min', ascending=False).astype(int)
                merged_df['Rank Diff'] = merged_df['CPRD Rank'] - merged_df['SAIL Rank']

        merged_df = merged_df.sort_values('Both %', ascending=False)
        # Format rank columns to be integers, filling NaNs with a placeholder
        rank_cols = ['CPRD Rank', 'SAIL Rank', 'Rank Diff']
        for col in rank_cols:
            # Use a placeholder for missing ranks, e.g., '-'
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').astype('Int64')

    return merged_df