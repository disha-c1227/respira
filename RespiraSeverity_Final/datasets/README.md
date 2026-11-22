# RespiraSeverity Datasets Documentation

This directory documents the multi-source dataset collections utilized in the RespiraSeverity framework.

## 1. Dataset Breakdown
Total validated cases in `manifest.csv`: **28,434**

| Target Disease Class | Case Count | Share (%) | Verified Sources |
| :--- | :--- | :--- | :--- |
| **Tuberculosis (TB)** | 10,641 | 37.42% | Chest X-Ray Dataset (muhammadrehan00) |
| **Normal** | 9,088 | 31.96% | Chest X-Ray Dataset (muhammadrehan00) |
| **Pneumonia** | 5,824 | 20.48% | Chest X-Ray Dataset (muhammadrehan00) |
| **COVID-19** | 2,881 | 10.13% | BIMCV (rafiko1) & RICORD (raddar) |
| **Total** | **28,434** | **100.00%** | Multi-Center Public Repositories |

## 2. COVID-19 Clinical Severity Ground-Truth Labels
Annotated cases: **760**
- **Mild:** 234 cases (30.79%)
- **Moderate:** 284 cases (37.37%)
- **Severe:** 242 cases (31.84%)

Splits (`data/severity_splits/`):
- `train.csv`: 532 cases (70%)
- `val.csv`: 114 cases (15%)
- `test.csv`: 114 cases (15%)

## 3. Storage Location
Original image files are preserved intact in the local KaggleHub cache directory:
`C:\Users\Disha C\.cache\kagglehub\datasets\`
- `rafiko1/bimcv-covid19-c-0`
- `muhammadrehan00/chest-xray-dataset/versions/1`
- `raddar/ricord-covid19-xray-positive-tests/versions/1`
