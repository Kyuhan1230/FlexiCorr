"""
Memory tracking utilities for adaptive correlation calculator.
메모리 사용량을 모니터링하고 추적하는 유틸리티 모듈입니다.
"""

import os
import psutil
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MemoryTracker:
    """시스템과 프로세스의 메모리 사용량을 추적하는 클래스"""
    
    def __init__(self):
        self._process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> float:
        """
        현재 프로세스의 메모리 사용량을 GB 단위로 반환합니다.
        
        Returns:
            float: 현재 메모리 사용량 (GB)
        """
        try:
            return self._process.memory_info().rss / (1024 ** 3)
        except Exception as e:
            logger.warning(f"메모리 사용량 측정 실패: {str(e)}")
            return 0.0
    
    def get_available_memory(self) -> float:
        """
        시스템의 사용 가능한 메모리를 GB 단위로 반환합니다.
        
        Returns:
            float: 사용 가능한 메모리 (GB)
        """
        try:
            return psutil.virtual_memory().available / (1024 ** 3)
        except Exception as e:
            logger.warning(f"가용 메모리 측정 실패: {str(e)}")
            return 0.0
    
    def estimate_memory_needs(self, rows: int, cols: int) -> Tuple[float, float]:
        """
        주어진 데이터 크기에 대한 예상 메모리 사용량을 계산합니다.
        
        Args:
            rows (int): 행 수
            cols (int): 열 수
            
        Returns:
            Tuple[float, float]: (최소 필요 메모리 GB, 권장 메모리 GB)
        """
        # 기본 데이터 크기 (8 bytes per float64)
        base_memory = (rows * cols * 8) / (1024 ** 3)
        
        # 상관행렬 크기
        corr_matrix_memory = (cols * cols * 8) / (1024 ** 3)
        
        # 최소 필요 메모리 (원본 데이터 + 상관행렬)
        min_memory = base_memory + corr_matrix_memory
        
        # 권장 메모리 (중간 계산 결과 고려)
        recommended_memory = min_memory * 2.5
        
        return min_memory, recommended_memory
    
    def check_memory_feasibility(
        self,
        rows: int,
        cols: int,
        memory_limit: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        주어진 데이터 크기에 대한 메모리 실행 가능성을 검사합니다.
        
        Args:
            rows (int): 행 수
            cols (int): 열 수
            memory_limit (float, optional): 사용할 최대 메모리 비율 (0.0 ~ 1.0)
            
        Returns:
            Tuple[bool, str]: (실행 가능 여부, 메시지)
        """
        min_memory, recommended_memory = self.estimate_memory_needs(rows, cols)
        available_memory = self.get_available_memory()
        
        if memory_limit:
            available_memory *= memory_limit
        
        if available_memory < min_memory:
            return False, f"필요 메모리({min_memory:.2f}GB)가 가용 메모리({available_memory:.2f}GB)를 초과합니다."
        
        if available_memory < recommended_memory:
            return True, f"메모리가 부족할 수 있습니다. 권장: {recommended_memory:.2f}GB, 가용: {available_memory:.2f}GB"
        
        return True, "충분한 메모리가 있습니다."

    def monitor_usage(self, callback=None):
        """
        메모리 사용량을 모니터링하고 선택적으로 콜백을 실행합니다.
        
        Args:
            callback (callable, optional): 메모리 사용량을 전달받을 콜백 함수
        """
        current_usage = self.get_memory_usage()
        available = self.get_available_memory()
        
        if callback:
            callback(current_usage, available)
        
        return current_usage, available
