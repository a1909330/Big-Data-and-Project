# Data-Driven Correctional Capacity Planning and Crime Forecasting

## Project Overview

This research project employs advanced machine learning techniques to address the critical challenge of correctional capacity planning and optimal facility location in South Australia. The study integrates SARIMA time series forecasting for temporal crime prediction with K-Means clustering for spatial crime pattern analysis to provide evidence-based insights for resource allocation and infrastructure planning.

**Author:** Sushant Randhawa (a1909330)  
**Institution:** University of Adelaide, School of Computer and Mathematical Sciences  
**Degree:** Master in Data Science  
**Date:** August 17, 2025

## Key Features

- **Dual-Model Approach**: Combines temporal forecasting (SARIMA) with spatial clustering (K-Means)
- **Comprehensive Data Processing**: Analysis of 72,636 individual criminal occurrences across 1,171 suburbs
- **Geospatial Analysis**: Interactive crime mapping and hotspot identification across 8 SA districts
- **Predictive Analytics**: 90-day crime forecasting with seasonal pattern recognition
- **Risk Classification**: Three-tier risk assessment (High/Medium/Low priority regions)
- **Infrastructure Planning**: Evidence-based recommendations for correctional facility placement

## Research Context

This project addresses South Australia's critical criminal justice challenges:
- **Prison Population Crisis**: 32% increase over the past decade (second-fastest growth in Australia)
- **Resource Constraints**: Police shortages requiring emergency staffing measures
- **Historic Investment**: $395 million government commitment to criminal justice infrastructure (2025-26 Budget)

The research provides data-driven spatial and temporal intelligence to optimize resource allocation and support the transition from reactive to predictive policing strategies.

## Requirements

### System Requirements
- **Python Version**: 3.9.7+
- **Hardware**: Intel Core i7 with 16GB RAM (or equivalent)
- **OS**: Compatible with Windows, macOS, and Linux

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn geopandas statsmodels zipfile
```

### Detailed Dependencies

- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical computing and statistical operations
- **matplotlib**: Data visualization and plotting
- **seaborn**: Statistical data visualization
- **scikit-learn**: K-Means clustering and StandardScaler preprocessing
- **geopandas**: Geospatial data analysis and mapping
- **statsmodels**: SARIMA time series modeling
- **zipfile**: Archive extraction and file management

## Dataset Information

### Primary Data Sources

1. **Crime Statistics Dataset**: 
   - Source: South Australia Police via Data SA Government Portal
   - Period: Q1-Q3 2024-25
   - Coverage: 72,636 individual criminal occurrences
   - Geographic Scope: 1,171 suburbs across 8 SA districts

2. **Geographic Reference Data**:
   - South Australian Outreach Reference Postcode (SAORP) mapping
   - Suburb boundaries dataset from Data SA Government Portal
   - Regional classification covering 8 SA districts

### Regional Coverage
- Northern Region
- York and Lower North Region  
- Eyre Region
- Outer Adelaide Region
- Kangaroo Island Region
- Adelaide Region
- Murray Lands Region
- South East Region

## Methodology

### 1. Data Preprocessing
- **Clustering Pipeline**: Regional aggregation, StandardScaler normalization, missing value handling
- **Time Series Pipeline**: Temporal formatting, date conversion, monthly resampling for forecasting

### 2. Spatial Analysis (K-Means Clustering)
- **Optimal Clusters**: k=3 determined through Elbow Method and Silhouette Analysis
- **Risk Categories**: 
  - High-risk: 2% of regions (18 urban hotspots)
  - Medium-risk: 11% of regions (132 regional hubs)
  - Low-risk: 87% of regions (1,021 suburban areas)
- **Geographic Coherence**: 73% of cluster members share regional borders

### 3. Temporal Forecasting (SARIMA)
- **Model Configuration**: SARIMA(1,1,1)(1,0,1,7) with weekly seasonality
- **Parameter Optimization**: Grid search across 324 combinations using AIC criterion
- **Performance Metrics**: MAE: 15.3, RMSE: 22.1, MAPE: 6.9%
- **Forecast Horizon**: 90-day predictions with 95% confidence intervals

## Key Research Findings

### Crime Distribution Patterns
- **High Concentration**: Adelaide accounts for 80% of all crimes (56,274 offenses)
- **Regional Disparity**: Only 2% of regions classified as high-crime requiring maximum security infrastructure
- **Seasonal Trends**: Predictable weekly patterns ranging from 175-300 incidents

### Infrastructure Recommendations
- **High-Priority Areas**: Immediate investment in high-security facilities for Adelaide metropolitan area
- **Medium-Priority Regions**: 2-3 strategically placed rehabilitation-focused facilities for regional hubs
- **Low-Priority Areas**: Cost-effective mobile services or shared regional facilities

### Operational Insights
- **Staffing Requirements**: Flexible models with 70% surge capacity over baseline operations
- **Resource Allocation**: Evidence-based framework for $395M infrastructure investment
- **Predictive Accuracy**: 94% pattern recognition accuracy for seasonal crime cycles

## Project Structure

```
project/
├── data/
│   ├── raw/                    # Original SA Police crime data
│   ├── processed/              # Cleaned datasets for analysis
│   └── geographic/             # SAORP mapping and suburb boundaries
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── spatial_clustering.ipynb
│   ├── time_series_forecasting.ipynb
│   └── results_integration.ipynb
├── src/
│   ├── clustering_analysis.py
│   ├── sarima_forecasting.py
│   └── visualization_utils.py
├── output/
│   ├── maps/                   # Geographic visualizations
│   ├── forecasts/              # SARIMA prediction plots
│   └── clusters/               # K-means results
├── docs/
│   ├── final_report.pdf
│   └── methodology.md
├── README.md
└── requirements.txt
```

## Usage

### Basic Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load SA Police crime data
crime_data = pd.read_csv('data/raw/sa_police_crime_q1_q3_2024_25.csv')

# Spatial clustering analysis
scaler = StandardScaler()
scaled_features = scaler.fit_transform(crime_features)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Time series forecasting
model = SARIMAX(daily_crimes, order=(1,1,1), seasonal_order=(1,0,1,7))
sarima_results = model.fit()
forecast = sarima_results.forecast(steps=90)

# Geographic visualization
crime_gdf = gpd.read_file('data/geographic/sa_regions.shp')
crime_gdf['cluster'] = clusters
crime_gdf.plot(column='cluster', cmap='viridis', legend=True)
```

### Model Performance Validation

```python
# SARIMA model evaluation
mae = mean_absolute_error(test_actual, test_predictions)  # 15.3
rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))  # 22.1
mape = mean_absolute_percentage_error(test_actual, test_predictions)  # 6.9%

# Clustering validation
silhouette_avg = silhouette_score(scaled_features, clusters)
```

## Results and Impact

### Academic Contributions
1. **Methodological Innovation**: First integrated dual-model system for simultaneous spatial-temporal correctional planning in Australia
2. **Performance Validation**: Superior forecasting accuracy compared to traditional ARIMA methods
3. **Big Data Processing**: Novel preprocessing pipelines for multi-source geographic and temporal data integration

### Practical Applications
- **Policy Framework**: Actionable three-tier risk classification system
- **Resource Optimization**: Evidence-based allocation of $395M infrastructure investment  
- **Operational Planning**: Transition from reactive to predictive policing strategies

## Limitations

- **Temporal Scope**: Limited to Q1-Q3 2024-25 data (restricts long-term trend analysis)
- **Feature Set**: Excludes socio-economic and demographic variables
- **Population Normalization**: Missing population data prevents crime rate per capita analysis
- **Forecast Horizon**: 90-day prediction limit for operational planning

## Future Research Directions

### Technical Enhancements
- **Extended Datasets**: Multi-year temporal analysis for annual cyclical patterns
- **Advanced ML Models**: Prophet and LSTM implementations for non-linear pattern detection
- **Feature Engineering**: Integration of socio-economic indicators and demographic data

### Policy Applications  
- **Cost-Benefit Analysis**: Integration of facility construction costs with public safety outcomes
- **Rehabilitation Metrics**: Community impact assessments and recidivism pattern analysis
- **Multi-Jurisdiction Scaling**: Framework adaptation for other Australian states

## Replication Package

### GitHub Repository
- **Full Project**: [a1909330/Big-Data-and-Project](https://github.com/a1909330/Big-Data-and-Project)
- **Code Access**: Complete data preprocessing scripts and analysis notebooks
- **Documentation**: Detailed methodology and implementation guides

### Overleaf Documentation  
- **Academic Paper**: [Complete project documentation](https://www.overleaf.com/project-link)
- **Reproducible Research**: Step-by-step replication instructions

## Citation

```bibtex
@mastersthesis{randhawa2025crime,
    title={Data-Driven Correctional Capacity Planning and Crime Forecasting},
    author={Randhawa, Sushant},
    year={2025},
    school={University of Adelaide},
    address={Adelaide, South Australia},
    type={Master of Data Science Thesis}
}
```

## Contact Information

- **Author**: Sushant Randhawa
- **Student ID**: a1909330  
- **Institution**: University of Adelaide, School of Computer and Mathematical Sciences
- **Location**: Adelaide, South Australia, Australia
- **Project Supervisor**: Dr. Hussain Ahmad

## Acknowledgments

Special thanks to:
- **Dr. Hussain Ahmad** - Project supervision and guidance
- **Manish and Haider Ali Lokhand** - Technical support and collaboration
- **University of Adelaide** - Research resources and infrastructure
- **South Australia Police** - Crime data provision through Data SA Portal
- **Data SA Government Portal** - Geographic reference datasets

---

*This project represents a significant contribution to evidence-based criminal justice policy in Australia, providing practical tools for optimizing correctional infrastructure investments while advancing academic knowledge in predictive criminology.*

**Last Updated**: August 17, 2025
