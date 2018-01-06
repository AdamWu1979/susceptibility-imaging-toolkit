import numpy as np

def parse_int(s):
    return np.int(s)

def parse_float(s):
    return np.float(s)

def parse_array(s, dtype):
    try: 
        beginning, bracket = s.index('['), '['
    except IndexError:
        try:
            beginning, bracket = s.index('('), '('
        except IndexError:
            beginning, bracket = s.index('{'), '{'
    if bracket == '[':
        end = s.index(']')
    elif bracket == '(':
        end = s.index(')')
    elif bracket == '{':
        end = s.index('}')
    if ',' in s:
        if ' ' in s:
            return np.array(s[beginning+1:end].split(', ')).astype(dtype)
        return np.array(s[beginning+1:end].split(',')).astype(dtype)
    return np.array(s[beginning+1:end].split(' ')).astype(dtype)
