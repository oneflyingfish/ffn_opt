import sys
import os

__INIT_THIRD_PARTY__ = False

if not __INIT_THIRD_PARTY__:
    __INIT_THIRD_PARTY__ = True

    import_paths = [".", "util_inference/src"]

    third_party_fold_path = os.path.dirname(os.path.abspath(__file__))

    # sys.path.insert(0, third_party_fold_path)

    for path in import_paths:
        path = path.strip()
        if path == "." or len(path) < 1:
            sys.path.insert(0, third_party_fold_path)
        else:
            sys.path.insert(0, os.path.join(third_party_fold_path, path))
    # print(sys.path)