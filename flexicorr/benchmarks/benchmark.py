"""
Benchmark module for adaptive correlation calculator
상관계수 계산 방법들의 성능을 비교하는 벤치마크 모듈
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from numba import jit
import time
import psutil
import os
from typing import Dict, List, Tuple
import json
from datetime import datetime
from ..calculator import CorrelationCalculator, CalculationMethod
from ..utils.memory_tracker import MemoryTracker

class BenchmarkResult:
    """벤치마크 결과를 저장하고 관리하는 클래스"""
    
    def __init__(self, data_size: Tuple[int, int], method: str):
        self.data_size = data_size
        self.method = method
        self.execution_time = 0.0
        self.memory_used = 0.0
        self.success = False
        self.error = None
    
    def to_dict(self) -> Dict:
        return {
            'data_size': f"{self.data_size[0]}×{self.data_size[1]}",
            'method': self.method,
            'execution_time': f"{self.execution_time:.3f}s",
            'memory_used': f"{self.memory_used:.2f}GB",
            'success': self.success,
            'error': str(self.error) if self.error else None
        }

def generate_test_data(rows: int, cols: int) -> pd.DataFrame:
    """테스트용 데이터 생성"""
    np.random.seed(42)  # 재현성을 위한 시드 설정
    return pd.DataFrame(
        np.random.randn(rows, cols),
        columns=[f'col_{i}' for i in range(cols)]
    )

def run_single_benchmark(
    data: pd.DataFrame,
    method: str,
    memory_tracker: MemoryTracker
) -> BenchmarkResult:
    """단일 방법에 대한 벤치마크 실행"""
    result = BenchmarkResult(data.shape, method)
    
    try:
        initial_memory = memory_tracker.get_memory_usage()
        start_time = time.time()
        
        if method == 'pandas':
            _ = data.corr()
        elif method == 'numpy':
            _ = np.corrcoef(data.values.T)
        elif method == 'dask':
            ddf = dd.from_pandas(data, npartitions=os.cpu_count())
            _ = ddf.corr().compute()
        elif method == 'adaptive':
            calculator = CorrelationCalculator()
            _ = calculator.calculate(data)[0]
        else:
            raise ValueError(f"Unknown method: {method}")
            
        result.execution_time = time.time() - start_time
        result.memory_used = memory_tracker.get_memory_usage() - initial_memory
        result.success = True
        
    except Exception as e:
        result.error = e
        result.success = False
        
    return result

def run_comprehensive_benchmark(
    save_results: bool = True,
    test_sizes: List[Tuple[int, int]] = None
) -> Dict:
    """
    종합적인 벤치마크 실행
    
    Args:
        save_results: 결과를 파일로 저장할지 여부
        test_sizes: 테스트할 데이터 크기 목록. None이면 기본값 사용
        
    Returns:
        Dict: 벤치마크 결과
    """
    if test_sizes is None:
        test_sizes = [
            (1_000, 10),
            (10_000, 20),
            (50_000, 50),
            (100_000, 100),
            (500_000, 100),
            (1_000_000, 100)
        ]
    
    memory_tracker = MemoryTracker()
    methods = ['pandas', 'numpy', 'dask', 'adaptive']
    all_results = {}
    
    for rows, cols in test_sizes:
        print(f"\nBenchmarking {rows:,} × {cols} dataset...")
        
        # 각 크기별로 새로운 데이터 생성
        data = generate_test_data(rows, cols)
        results = {}
        
        for method in methods:
            print(f"Testing {method}...")
            result = run_single_benchmark(data, method, memory_tracker)
            results[method] = result.to_dict()
        
        all_results[f"{rows}×{cols}"] = results
        
        # 메모리 정리
        del data
        import gc
        gc.collect()
    
    if save_results:
        save_benchmark_results(all_results)
    
    return all_results

def save_benchmark_results(results: Dict):
    """벤치마크 결과를 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"
    
    # docs 디렉토리가 없으면 생성
    os.makedirs('docs', exist_ok=True)
    filepath = os.path.join('docs', filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {filepath}")

def format_results(results: Dict) -> Tuple[List[str], List[List]]:
    """결과를 표 형식으로 포맷팅"""
    headers = ['Data Size', 'Pandas', 'NumPy', 'Dask', 'Adaptive']
    rows = []
    
    for size, size_results in results.items():
        row = [size]
        for method in ['pandas', 'numpy', 'dask', 'adaptive']:
            result = size_results[method]
            if result['success']:
                row.append(f"{result['execution_time']} ({result['memory_used']})")
            else:
                row.append("Failed")
        rows.append(row)
    
    return headers, rows

def print_system_info():
    """시스템 정보 출력"""
    print("\nSystem Information:")
    print("-" * 50)
    print(f"CPU Count: {os.cpu_count()}")
    print(f"Total Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Python Version: {platform.python_version()}")
    print(f"Operating System: {platform.system()} {platform.version()}")
    print("-" * 50)

if __name__ == "__main__":
    import platform
    from tabulate import tabulate
    
    print_system_info()
    
    # 벤치마크 실행
    results = run_comprehensive_benchmark()
    
    # 결과 출력
    headers, rows = format_results(results)
    print("\nBenchmark Results:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))