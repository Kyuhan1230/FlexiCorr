"""
Flexible Correlation Calculator
-----------------------------
데이터셋 크기에 따라 최적의 상관계수 계산 방법을 선택하는 모듈입니다.
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from numba import jit
import logging
from typing import Union, Tuple, Dict, Optional
from enum import Enum
import time
from .utils.memory_tracker import MemoryTracker

logger = logging.getLogger(__name__)

class CalculationMethod(Enum):
    """상관계수 계산 방법 열거형"""
    AUTO = 'auto'
    PANDAS = 'pandas'
    NUMPY = 'numpy'
    DASK = 'dask'
    NUMBA = 'numba'

@jit(nopython=True, parallel=True)
def _numba_corrcoef(data: np.ndarray) -> np.ndarray:
    """
    Numba로 최적화된 상관계수 계산
    
    Args:
        data (np.ndarray): 입력 데이터 배열
        
    Returns:
        np.ndarray: 상관계수 행렬
    """
    nrows, ncols = data.shape
    corr = np.empty((ncols, ncols))
    
    # 데이터 정규화
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    normalized = (data - mean) / std
    
    # 상관계수 계산
    for i in range(ncols):
        for j in range(i, ncols):
            c = (normalized[:,i] * normalized[:,j]).sum() / (nrows - 1)
            corr[i,j] = c
            corr[j,i] = c
    
    return corr

class CorrelationConfig:
    """상관계수 계산 설정"""
    def __init__(
        self,
        chunk_size: str = '64MB',
        memory_limit: float = 0.75,
        min_rows_for_dask: int = 50000,
        min_cols_for_dask: int = 50,
        n_workers: Optional[int] = None
    ):
        self.chunk_size = chunk_size
        self.memory_limit = memory_limit
        self.min_rows_for_dask = min_rows_for_dask
        self.min_cols_for_dask = min_cols_for_dask
        self.n_workers = n_workers

class CorrelationCalculator:
    """적응형 상관계수 계산기"""
    
    def __init__(self, config: Optional[CorrelationConfig] = None):
        """
        계산기 초기화
        
        Args:
            config (CorrelationConfig, optional): 계산 설정
        """
        self.config = config or CorrelationConfig()
        self.memory_tracker = MemoryTracker()
    
    def calculate(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        method: Union[str, CalculationMethod] = CalculationMethod.AUTO
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        상관계수 계산 실행
        
        Args:
            data: 입력 데이터 (DataFrame 또는 ndarray)
            method: 계산 방법 (기본값: AUTO)
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (상관계수 행렬, 메타데이터)
        """
        start_time = time.time()
        initial_memory = self.memory_tracker.get_memory_usage()
        
        # 입력 데이터 검증 및 변환
        df = self._validate_and_convert_input(data)
        rows, cols = df.shape
        
        # 메서드 선택
        if isinstance(method, str):
            method = CalculationMethod(method)
            
        if method == CalculationMethod.AUTO:
            method = self._select_method(rows, cols)
        
        logger.info(f"Selected method: {method.value}")
        
        try:
            # 메모리 가용성 검사
            feasible, msg = self.memory_tracker.check_memory_feasibility(
                rows, cols, self.config.memory_limit
            )
            if not feasible:
                logger.warning(msg)
                if method != CalculationMethod.DASK:
                    method = CalculationMethod.DASK
                    logger.info("Switched to DASK due to memory constraints")
            
            # 상관계수 계산
            correlation = self._calculate_with_method(df, method)
            
            # 메타데이터 준비
            meta = {
                'rows': rows,
                'columns': cols,
                'method_used': method.value,
                'processing_time_sec': time.time() - start_time,
                'memory_usage_gb': self.memory_tracker.get_memory_usage() - initial_memory,
                'memory_message': msg
            }
            
            return self._prepare_result(correlation, meta)
            
        except Exception as e:
            logger.error(f"Calculation failed with {method.value}: {str(e)}")
            return self._try_fallback_methods(df)
    
    def _validate_and_convert_input(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """입력 데이터 검증 및 변환"""
        if isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            return data.copy()
        else:
            raise ValueError("Input must be DataFrame or ndarray")
    
    def _select_method(self, rows: int, cols: int) -> CalculationMethod:
        """최적의 계산 방법 선택"""
        if rows < self.config.min_rows_for_dask and cols < self.config.min_cols_for_dask:
            return CalculationMethod.PANDAS
        
        # 메모리 기반 선택
        min_memory, _ = self.memory_tracker.estimate_memory_needs(rows, cols)
        available_memory = self.memory_tracker.get_available_memory()
        
        if min_memory > available_memory * self.config.memory_limit:
            return CalculationMethod.DASK
        elif rows < 100000:
            return CalculationMethod.NUMPY
        else:
            return CalculationMethod.DASK
    
    def _calculate_with_method(
        self,
        df: pd.DataFrame,
        method: CalculationMethod
    ) -> pd.DataFrame:
        """선택된 방법으로 상관계수 계산"""
        if method == CalculationMethod.PANDAS:
            return df.corr()
            
        elif method == CalculationMethod.NUMPY:
            return pd.DataFrame(
                np.corrcoef(df.values.T),
                index=df.columns,
                columns=df.columns
            )
            
        elif method == CalculationMethod.NUMBA:
            return pd.DataFrame(
                _numba_corrcoef(df.values),
                index=df.columns,
                columns=df.columns
            )
            
        elif method == CalculationMethod.DASK:
            ddf = dd.from_pandas(df, npartitions=self.config.n_workers or 'auto')
            return ddf.corr().compute()
            
        else:
            raise ValueError(f"Unknown method: {method.value}")
    
    def _try_fallback_methods(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """폴백 방법 시도"""
        fallback_methods = [
            CalculationMethod.DASK,
            CalculationMethod.NUMPY,
            CalculationMethod.PANDAS
        ]
        
        for method in fallback_methods:
            try:
                logger.info(f"Trying fallback method: {method.value}")
                return self.calculate(df, method)
            except Exception as e:
                logger.warning(f"Fallback method {method.value} failed: {str(e)}")
        
        raise RuntimeError("All correlation calculation methods failed")
    
    def _prepare_result(
        self,
        correlation: pd.DataFrame,
        meta: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """결과 준비 및 후처리"""
        # NaN 처리
        correlation = correlation.fillna(0)
        
        # 대칭성 확인 및 보정
        if not np.allclose(correlation, correlation.T):
            correlation = (correlation + correlation.T) / 2
        
        return correlation, meta
