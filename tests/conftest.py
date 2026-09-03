"""테스트는 CPU JAX 로 고정: float64 단일 궤적 시뮬레이션은 GPU 보다 CPU 가 빠르다 (Linux RTX 5090: 23 s vs 74 s). 원하면 JAX_PLATFORMS=cuda 로 덮어쓴다."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
