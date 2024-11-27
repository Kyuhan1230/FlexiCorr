"""Unit tests for Flexible correlation calculator."""
import pytest
import numpy as np
import pandas as pd
from flexicorr import CorrelationCalculator
from flexicorr.utils.memory_tracker import MemoryTracker

@pytest.fixture
def small_dataset():
    """Create a small test dataset."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(1000, 10),
        columns=[f'col_{i}' for i in range(10)]
    )

@pytest.fixture
def medium_dataset():
    """Create a medium test dataset."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(50000, 50),
        columns=[f'col_{i}' for i in range(50)]
    )

class TestCorrelationCalculator:
    def test_initialization(self):
        """Test calculator initialization."""
        calculator = CorrelationCalculator()
        assert calculator is not None
        assert hasattr(calculator, 'config')
    
    def test_small_dataset(self, small_dataset):
        """Test correlation calculation with small dataset."""
        calculator = CorrelationCalculator()
        correlation, meta = calculator.calculate(small_dataset)
        
        # 기본 검증
        assert isinstance(correlation, pd.DataFrame)
        assert correlation.shape == (10, 10)
        assert meta['method_used'] == 'pandas'
        
        # pandas의 corr()과 결과 비교
        pd.testing.assert_frame_equal(
            correlation,
            small_dataset.corr(),
            check_exact=False,
            rtol=1e-5
        )
    
    def test_medium_dataset(self, medium_dataset):
        """Test correlation calculation with medium dataset."""
        calculator = CorrelationCalculator()
        correlation, meta = calculator.calculate(medium_dataset)
        
        assert isinstance(correlation, pd.DataFrame)
        assert correlation.shape == (50, 50)
        assert meta['method_used'] in ['numpy', 'dask']
        
    def test_invalid_input(self):
        """Test calculator with invalid input."""
        calculator = CorrelationCalculator()
        
        with pytest.raises(ValueError):
            calculator.calculate(None)
        
        with pytest.raises(ValueError):
            calculator.calculate([1, 2, 3])
    
    def test_memory_tracking(self, medium_dataset):
        """Test memory tracking functionality."""
        tracker = MemoryTracker()
        initial_memory = tracker.get_memory_usage()
        
        calculator = CorrelationCalculator()
        _, meta = calculator.calculate(medium_dataset)
        
        assert 'memory_usage_gb' in meta
        assert meta['memory_usage_gb'] > initial_memory
    
    def test_correlation_values(self):
        """Test correlation values for known cases."""
        # 완벽한 상관관계 케이스
        data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10]  # a의 2배
        })
        
        calculator = CorrelationCalculator()
        correlation, _ = calculator.calculate(data)
        
        assert correlation.iloc[0, 1] == pytest.approx(1.0)
        
    def test_nan_handling(self):
        """Test handling of NaN values."""
        data = pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5],
            'b': [2, 4, 6, np.nan, 10]
        })
        
        calculator = CorrelationCalculator()
        correlation, _ = calculator.calculate(data)
        
        assert not correlation.isna().any().any()

if __name__ == '__main__':
    pytest.main([__file__])
