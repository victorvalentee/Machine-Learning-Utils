"""
Data cleaning is the process of detecting and correcting
(or removing) corrupt or inaccurate records from a dataset.
"""


def strip_whitespaces(df):
    """
    Strip whitespaces from every string column of the dataframe.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    DataFrame
        A copy of `df` without the affected rows.
    """
    
    return df.applymap(
        # applymap: apply function to every column of the dataframe.
        lambda x: x.strip() if isinstance(x, str) else x
    )


def drop_na_rows(df, col_names=None):
    """
    Drop rows from `df[col_names]` that contain missing values.

    Parameters
    ----------
    df : DataFrame
    col_names : list (optional)
        Names of the columns to be searched for NaN values.

    Returns
    -------
    DataFrame
        A copy of `df` without the affected rows.
    """

    return df.dropna(axis=0, how='any', subset=col_names)


def keep_rows_by_filter(df, col_name, keep_list):
    """
    Keep rows from `df[col_name]` that contain any of the values in the `keep_list`
    and drop everything else.

    Parameters
    ----------
    df : DataFrame
    col_name : str
        Name of the column to be filtered.
    keep_list : list
        List containing the values that we want in our rows.

    Returns
    -------
    DataFrame
        A copy of the resulting dataframe.
    """

    # Make query with filtering conditions.
    query_str = ""

    for value in drop_list:
        if query_str: query_str += " | "
        query_str += f"{col_name} == @value"

    # Return filtered dataframe.
    return df.query(query_str)


def drop_rows_by_filter(df, col_name, drop_list):
    """
    Drop rows from `df[col_name]` that contain any of the values in the `drop_list`
    and keep everything else.

    Parameters
    ----------
    df : DataFrame
    col_name : str
        Name of the column to be filtered.
    drop_list : list
        List containing the values that we don't want in our rows.

    Returns
    -------
    DataFrame
        A copy of the resulting dataframe.
    """

    # Make query with filtering conditions.
    query_str = ""

    for value in drop_list:
        if query_str: query_str += " & "
        query_str += f"{col_name} != @value"

    # Return filtered dataframe.
    return df.query(query_str)


def df_find_replace(df, col_name, find, replace):
    """
    Find string `find` in `df[col_name]` and replace it for `replace_str`.

    Parameters
    ----------
    df : DataFrame
    col_name : str
        Name of the column of the dataframe that contains empty strings.

    Returns
    -------
    DataFrame
        A copy of `df` with the values.
    """
    
    return df[col_name] = df[col_name].str.replace(find, replace)