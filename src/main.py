from src.common.utils import load_params
from src.common.utils import init_ont

def main():
    keys = load_params(path="config/keys.yaml")
    init_ont(key=keys["ont_key"])
    pass

if __name__ == "__main__":
    main()