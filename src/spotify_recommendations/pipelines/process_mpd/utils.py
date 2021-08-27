def str2bool(v):  # Note the self to allow it to be in a class
    return v.lower() in ('yes', 'true', 't', '1', 'yea', 'verily')  # lower() is a method


def convert_text_bool_to_python_bool(val):
    return str2bool(val)


def chunk(data, n):
    return [data[x: x + n] for x in range(0, len(data), n)]


def str2bool(v):  # Note the self to allow it to be in a class
    return v.lower() in ('yes', 'true', 't', '1', 'yea', 'verily')  # lower() is a method


def convert_text_bool_to_python_bool(val):
    return str2bool(val)
