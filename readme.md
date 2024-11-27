# FlexiCorrelation

데이터셋 크기에 따라 자동으로 최적의 상관계수 계산 방법을 선택하는 Python 패키지입니다.

## 개요

대규모 데이터셋의 상관계수를 계산할 때, 데이터 크기에 따라 적절한 계산 방법을 선택하는 것이 중요합니다. 이 패키지는 데이터셋의 크기를 분석하여 자동으로 가장 효율적인 계산 방법을 선택하고 실행합니다.

## 배경
이 레포지토리는 파이썬 패키지 개발 학습 과정의 결과물이면서, 
실제 업무 환경에서 마주치는 문제를 해결하고자 하는 시도입니다. 

GPU가 없는 일반적인 업무용 컴퓨터에서도 대용량 데이터의 상관관계를 효율적으로 계산할 수 있는 방법을 고민하였고, 
그 결과로 데이터 크기에 따라 최적의 계산 방법을 자동으로 선택하는 접근법을 채택하였습니다.

완벽한 해결책은 아닐 수 있지만, 실무에서 마주하는 제약 조건 속에서 최선의 성능을 끌어내고자 하였습니다.<br>
이 과정에서 패키지 구조 설계, 문서화, 테스트 작성, 성능 측정 등 파이썬 패키지 개발의 전반적인 과정을 경험할 수 있었습니다.

### CPU 기반 구현에 대하여
이 패키지는 의도적으로 CPU 기반으로 구현되었습니다. 그 이유는 다음과 같습니다:
1. 범용성: GPU가 없는 환경에서도 실행 가능
2. 안정성: CUDA 버전 호환성 문제 없음
3. 배포 용이성: 추가 드라이버나 CUDA 설치 불필요
4. 비용 효율성: GPU 서버 없이도 활용 가능

필요한 경우 CuPy를 사용한 GPU 가속은 선택적으로 활성화할 수 있도록 향후 확장할 예정입니다.

## 요구사항

### Python 버전
- Python >= 3.8

### 주요 라이브러리 버전
- pandas >= 1.3.0
- numpy >= 1.20.0
- dask >= 2021.6.0
- numba >= 0.53.0
- psutil >= 5.8.0

## 설치

```bash
pip install flexicorr
```

또는 소스에서 직접 설치:

```bash
git clone https://github.com/kyuhan1230/FlexiCorr.git
cd FlexiCorr
pip install -e .
```

## 사용법

```python
from flexicorr import CorrelationCalculator

# 계산기 초기화
calculator = CorrelationCalculator()

# 데이터프레임에 대한 상관계수 계산
correlation, meta = calculator.calculate(your_dataframe)

# 메타 정보 확인
print(f"Used method: {meta['method_used']}")
print(f"Processing time: {meta['processing_time_sec']:.2f} seconds")
```

## 벤치마크 결과

다양한 데이터셋 크기에 대한 성능 비교:

| 데이터 크기 (rows × cols) | Pandas | NumPy | Dask | Numba | FlexiCorr | 최적 선택 |
|---------------------------|--------|-------|------|-------|----------|-----------|
| 1,000 × 10 | 0.002s | 0.005s | 0.2s | 0.4s | 0.003s | Pandas |
| 10,000 × 20 | 0.01s | 0.015s | 0.3s | 0.5s | 0.012s | Pandas |
| 50,000 × 50 | 0.15s | 0.08s | 0.4s | 0.2s | 0.085s | NumPy |
| 100,000 × 100 | 0.8s | 0.3s | 0.5s | 0.4s | 0.5s | Dask |
| 500,000 × 100 | 4.5s* | 2.1s* | 1.2s | 1.8s* | 1.3s | Dask |
| 1,000,000 × 100 | 실패** | 실패** | 2.5s | 실패** | 2.6s | Dask |

\* : 높은 메모리 사용량
\** : 메모리 부족으로 실행 실패

자세한 벤치마크 결과는 [benchmark_results.md](docs/benchmark_results.md)를 참조하세요.

## 테스트 결과

모든 테스트는 다음 환경에서 수행되었습니다:
- CPU: Intel Core i7-9700K
- RAM: 32GB DDR4
- OS: Ubuntu 20.04 LTS
- Python 3.8.10

테스트 실행 방법:
```bash
pytest tests/
```

자세한 테스트 결과는 [test_results.md](docs/test_results.md)를 참조하세요.

## 기여하기

버그 리포트, 기능 제안, 풀 리퀘스트 모두 환영합니다.

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
