"""
Flexible Correlation Calculator
-----------------------------
데이터셋 크기에 따라 최적의 상관계수 계산 방법을 자동으로 선택하는 패키지
"""

from .calculator import CorrelationCalculator, CorrelationConfig, CalculationMethod
from .utils.memory_tracker import MemoryTracker

__version__ = '0.1.0'
__author__ = 'KyuHan,Seok'
__email__ = 'asdm159@gmail.com'

__all__ = [
    'CorrelationCalculator',
    'CorrelationConfig',
    'CalculationMethod',
    'MemoryTracker',
]