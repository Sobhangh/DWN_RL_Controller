import numpy as np

def produce_min_max_map(df, numeric_cols):
    min_max_map = {}
    #print(df)
    for col in numeric_cols:
        col_min = np.nanmin(df[col])
        col_max = np.nanmax(df[col])
        # assert thast both are not nan
        assert not(math.isnan(col_min)), f"col: {col}, min: {col_min}, uniques: {df[col].unique()}"
        assert not(math.isnan(col_max))
        min_max_map[col] = (col_min, col_max)
    return min_max_map

def produce_categorical_map(df, cat_cols):
    cat_unique_map = {}
    for col in cat_cols:
        cat_unique_map[col] = sorted(df[col].unique())
    return cat_unique_map


import math
def encode_numeric(value, col, map, max_encoding_size=10,isFloat=False):
    """
    Given a numeric value and its column index, produce a 'thermometer' 
    encoding with an adjustable maximum size (max_encoding_size).
    If the distinct integer range for this col exceeds max_encoding_size, 
    we compress the range into max_encoding_size buckets.
    
    Example: 
        If min_val=0, max_val=9, distinct_range=10 <= max_encoding_size=20,
        length = 10, 
        value=2 -> [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    
    If min_val=0, max_val=100, distinct_range=101 > 20,
    length = 20, 
    step = 101 / 20 = 5.05
    value=10 -> offset = floor((10 - 0)/5.05) = floor(1.98) = 1
              -> [1, 0, 0, 0, ..., 0] (20 bits total)
    """
    min_val, max_val = map[col]
    distinct_range = int((max_val - min_val) + 1)  # total possible integer values
    #print(distinct_range)
    # Determine the thermometer length (L).
    length = min(distinct_range, max_encoding_size)

    # step indicates how many raw integer values per 1 increment in the thermometer
    step = distinct_range / float(length)

    # offset = how many bits should be '1'
    offset = int((value - min_val) / step)
    
    # Clamp offset to avoid going out of bounds (e.g., if value == max_val)
    if offset >= length:
        offset = length - 1

    # Build the thermometer vector
    # print(offset, length)
    thermometer = [1]*offset + [0]*(length - offset)
    return thermometer

def encode_categorical(value, col, map):
    """
    Given a category value and its column index,
    produce a one-hot encoding using cat_unique_map.
    """
    categories = map[col]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return [0] * len(categories)
    
    return [1 if value == cat else 0 for cat in categories]

def encode_row(row, numeric_cols, cat_unique_map, min_max_map,
               cat_info, 
               num_info,
               total_dim,
               max_encoding_size=10, columns=20,
               onlyFloat=False):
    """
    Encode a single row of the original DataFrame using:
      - Thermometer encoding for numeric columns (up to max_encoding_size bits)
      - One-hot encoding for categorical columns
    Returns a single list of binary features.
    """
    encoded = []
    if type(columns) == int:
        for col in range(columns):
            if col in numeric_cols:
                encoded.extend(encode_numeric(row[col], col, min_max_map, max_encoding_size, isFloat=onlyFloat))

            else:
                encoded.extend(encode_categorical(row[col], col, cat_unique_map))
    else:
        encoded = np.zeros(int(total_dim))
        for col in columns:
            if col in numeric_cols:
                matches = [d for d in num_info if d.get('col') ==col]
                if matches:
                    start = int(matches[0]['start'])
                    length = int(matches[0]['length'])
                    #print(start, length)
                else:
                    assert False
                # look where to place in num_info
                #print(num_info)
                #print(row)
                #print(encode_numeric(row[col], col, min_max_map, max_encoding_size))
                encoded[start:start+length] = encode_numeric(row[col], col, min_max_map, max_encoding_size, isFloat=onlyFloat)
                #print(encoded)
            else:
                matches = [d for d in cat_info if d.get('col') ==col]
                if matches:
                    start = int(matches[0]['start'])
                    length = int(matches[0]['length'])
                    #print(start, length)
                    assert len(matches) == 1
                else:
                    assert False
                
                encoded[start:start+length] = encode_categorical(row[col], col, cat_unique_map)
    return encoded


def build_categorical_numeric_info(cat, numerical, max_encoding_size=10, columns=None, onlyFloat=False):
    """
    Build 'categorical_info' and 'numeric_info' for all columns mentioned in 
    'cat' or 'numerical', assigning each column a (start, length) slice in the 
    overall input bit-vector.

    :param cat: dict { col_idx: list_of_categories }, e.g.
                 { 0: ['A11', 'A12', 'A13', 'A14'],
                   2: ['A30', 'A31', 'A32', 'A33', 'A34'], ... }
    :param numerical: dict { col_idx: (min_val, max_val) }, e.g.
                      { 1: (4,72), 4: (250,18424), ... }
    :param max_encoding_size: int that caps the thermometer length for numeric columns.
    :return: (categorical_info, numeric_info, total_dim)

      - categorical_info: a list of dicts, each like:
          { "col": col_idx, "start": <int>, "length": <int> }
      - numeric_info: a list of dicts, each like:
          { "col": col_idx, "start": <int>, "length": <int>, "thermometer": True }
      - total_dim: total number of bits used (i.e. the 'max' index in the input space).
    """
    categorical_info = []
    numeric_info = []

    current_index = 0
    
    # Gather all columns from either dict, then sort by col index
    #all_cols = sorted(set(cat.keys()).union(numerical.keys()))
    
    for col in columns:
        if col in cat:
            # Categorical column
            categories = cat[col]
            length = len(categories)  # one-hot length = #categories
            categorical_info.append({
                "col": col,
                "start": int(current_index),
                "length": int(length)
            })
            current_index += length
        elif col in numerical:
            # Numeric column
            min_val, max_val = numerical[col]
            if onlyFloat:
                distinct_range = max_encoding_size
            else:
                distinct_range = (max_val - min_val) + 1
            length = min(distinct_range, max_encoding_size)
            numeric_info.append({
                "col": col,
                "start": int(current_index),
                "length": int(length),
                "thermometer": True  # so we can interpret in a domain-constraint function
            })
            current_index += length

    total_dim = current_index
    return categorical_info, numeric_info, total_dim
