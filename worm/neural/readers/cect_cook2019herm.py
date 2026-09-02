"""c302용 Cook 2019 (자웅동체) 커넥톰 리더 래퍼.
c302.load_data_reader 는 이름에 'cect'가 있으면 get_instance() 를 호출하고 read_data(include_nonconnected_cells=True)
를 부르는데, cect의 Cook2019 리더는 이 인자를 받지 않아 어댑터가 필요하다.
사용: --reader worm.neural.readers.cect_cook2019herm
"""
from cect.readers.Cook2019HermReader import get_instance as _get_instance

class _Adapter:
    def __init__(self, inst):
        self._i = inst
    def read_data(self, include_nonconnected_cells=False):
        f = getattr(self._i, "read_data", None) or self._i._read_data
        return f()
    def read_muscle_data(self):
        f = getattr(self._i, "read_muscle_data", None) or self._i._read_muscle_data
        return f()
    def __getattr__(self, k):
        return getattr(self._i, k)

def get_instance(from_cache=True):
    return _Adapter(_get_instance(from_cache))
