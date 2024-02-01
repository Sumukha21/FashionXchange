import importlib


# Copied from https://github.com/justinpinkney/stable-diffusion/blob/main/ldm/util.py
def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# Copied from https://github.com/justinpinkney/stable-diffusion/blob/main/ldm/util.py
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def text_file_reader(file_path):
    """
    Read the contents of a text file.
    Args:
        file_path (str): The path to the text file.
    Returns:
        list: A list containing the lines of text from the file.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return lines
    
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
