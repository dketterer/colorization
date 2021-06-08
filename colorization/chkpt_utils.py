import glob
import os
import re
from typing import List, Tuple, Union


def get_chkpts_all(path: str, file_extension='.pth') -> List[Tuple[str, int]]:
    pat = r'.*-([0-9]*)' + file_extension
    base, ext = os.path.splitext(path)
    assert ext == file_extension

    results = []

    for file in glob.glob(f'{base}-[0-9]*{file_extension}'):
        if os.path.isfile(file):
            i = int(re.match(pat, file)[1])
            results.append((file, i))

    results.sort(key=lambda item: item[1], reverse=True)

    return results


def get_latest(path: str, file_extension='.pth') -> Union[str, None]:
    chkpts = get_chkpts_all(path, file_extension)
    if chkpts:
        latest_path, latest_iter = chkpts[0]
        return latest_path
    elif os.path.isfile(path):
        return path
    else:
        return None


def build_model_file_name(generic_name: str, iter: int) -> str:
    base, ext = os.path.splitext(generic_name)
    assert ext != ''
    return f'{base}-{iter:03d}{ext}'


def get_older_then_n(base_path: str, n: int, file_extension='.pth') -> List[str]:
    chkpts = get_chkpts_all(base_path, file_extension)

    if len(chkpts) > n:
        return [p for p, _ in chkpts[n:]]
    return []


def delete_older_then_n(base_path: str, keep_last_n=10):
    for p in get_older_then_n(base_path, keep_last_n):
        os.remove(p)
        assert not os.path.isfile(p)
