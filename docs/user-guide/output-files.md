# Output Files

DisruptSC generates comprehensive output data for analysis and visualization. This guide explains all output files and how to use them.

## Output Structure

Results are saved in timestamped directories:

```
output/<scope>/<timestamp>/
├── Time Series Data
│   ├── firm_data.json
│   ├── household_data.json
│   ├── country_data.json
│   └── flow_df_*.csv
├── Spatial Data
│   ├── firm_table.geojson
│   ├── household_table.geojson
│   └── transport_edges_with_flows_*.geojson
├── Network Data
│   ├── sc_network_edgelist.csv
│   └── io_table.csv
├── Analysis Results
│   ├── loss_per_country.csv
│   ├── loss_per_region_sector_time.csv
│   ├── loss_summary.csv
│   └── sensitivity_*.csv          # Sensitivity analysis only
└── Metadata
    ├── parameters.yaml
    └── exp.log
```

## Time Series Data

### Firm Data (`firm_data.json`)

Agent-level time series for all firms.

#### Structure
```json
{
  "production": {
    "0": {"1001": 500000, "1002": 750000, "1003": 2000000},
    "1": {"1001": 480000, "1002": 720000, "1003": 1900000},
    "2": {"1001": 450000, "1002": 690000, "1003": 1800000}
  },
  "inventory": {
    "AGR_KHM": {
      "0": {"1001": 50000, "1002": 75000},
      "1": {"1001": 45000, "1002": 70000}
    }
  },
  "finance": {
    "0": {"1001": 100000, "1002": 150000, "1003": 400000}
  }
}
```

#### Key Variables

| Variable | Description | Units |
|----------|-------------|-------|
| `production` | Output produced per time step | Model currency |
| `production_target` | Planned production | Model currency |
| `product_stock` | Finished goods inventory | Model currency |
| `inventory` | Input inventories by sector | Model currency |
| `purchase_plan` | Planned purchases by supplier | Model currency |
| `finance` | Financial position | Model currency |
| `profit` | Profit/loss per time step | Model currency |

#### Analysis Examples

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load firm data
with open('firm_data.json', 'r') as f:
    firm_data = json.load(f)

# Convert to DataFrame
production_df = pd.DataFrame(firm_data['production']).T
production_df.index = production_df.index.astype(int)

# Plot total production over time
total_production = production_df.sum(axis=1)
total_production.plot(title='Total Production Over Time')
plt.ylabel('Production (Model Currency)')
plt.xlabel('Time Step')
plt.show()

# Calculate production losses
baseline = total_production.iloc[0]
losses = baseline - total_production
cumulative_loss = losses.cumsum()

print(f"Peak loss: {losses.max():.0f}")
print(f"Total cumulative loss: {cumulative_loss.iloc[-1]:.0f}")
```

### Household Data (`household_data.json`)

Consumption patterns and welfare impacts.

#### Structure
```json
{
  "consumption": {
    "0": {"hh_1": 15000, "hh_2": 22000},
    "1": {"hh_1": 14500, "hh_2": 21000}
  },
  "spending": {
    "0": {"hh_1": 15000, "hh_2": 22000},
    "1": {"hh_1": 15200, "hh_2": 22500}
  },
  "consumption_loss": {
    "0": {"hh_1": 0, "hh_2": 0},
    "1": {"hh_1": 500, "hh_2": 1000}
  }
}
```

#### Key Variables

| Variable | Description | Units |
|----------|-------------|-------|
| `consumption` | Goods/services consumed | Model currency |
| `spending` | Total expenditure | Model currency |
| `consumption_loss` | Unmet demand | Model currency |
| `extra_spending` | Price increase impacts | Model currency |

### Country Data (`country_data.json`)

International trade impacts.

#### Structure
```json
{
  "exports": {
    "0": {"CHN": 500000, "THA": 300000},
    "1": {"CHN": 480000, "THA": 290000}
  },
  "imports": {
    "0": {"CHN": 200000, "THA": 150000}
  }
}
```

### Transport Flows (`flow_df_*.csv`)

Detailed transport flow data by time step.

```csv
edge_id,from_node,to_node,transport_mode,flow_volume,flow_value,travel_time
edge_001,node_123,node_456,roads,1500,750000,2.5
edge_002,node_456,node_789,roads,800,400000,1.8
edge_003,port_001,port_002,maritime,5000,2500000,24.0
```

#### Columns

| Column | Description | Units |
|--------|-------------|-------|
| `edge_id` | Transport link identifier | - |
| `from_node` | Origin node | - |
| `to_node` | Destination node | - |
| `transport_mode` | Mode of transport | - |
| `flow_volume` | Physical flow volume | Tons |
| `flow_value` | Economic value | Model currency |
| `travel_time` | Transit time | Hours |

## Spatial Data

### Firm Locations (`firm_table.geojson`)

Spatial firm data with attributes and results.

```json
{
  "type": "Feature",
  "properties": {
    "pid": 1001,
    "sector": "AGR_KHM",
    "region": "Phnom_Penh",
    "eq_production": 500000,
    "final_production": 450000,
    "production_loss": 50000,
    "importance": 0.15
  },
  "geometry": {
    "type": "Point",
    "coordinates": [-104.866, 11.555]
  }
}
```

#### Key Attributes

| Attribute | Description | Type |
|-----------|-------------|------|
| `pid` | Firm identifier | integer |
| `sector` | Region_sector code | string |
| `eq_production` | Baseline production | float |
| `final_production` | End production | float |
| `production_loss` | Total impact | float |
| `importance` | Economic importance | float |

### Household Locations (`household_table.geojson`)

Spatial household data with consumption impacts.

```json
{
  "properties": {
    "pid": "hh_1",
    "region": "Phnom_Penh", 
    "population": 15000,
    "baseline_consumption": 150000,
    "final_consumption": 145000,
    "consumption_loss": 5000,
    "extra_spending": 2000
  }
}
```

### Transport Flows (`transport_edges_with_flows_*.geojson`)

Network visualization with flow data.

```json
{
  "type": "Feature",
  "properties": {
    "edge_id": "edge_001",
    "highway": "primary",
    "length_km": 15.2,
    "baseline_flow": 1500,
    "disrupted_flow": 1200,
    "flow_reduction": 300,
    "utilization": 0.75
  },
  "geometry": {
    "type": "LineString",
    "coordinates": [[-104.5, 11.1], [-104.4, 11.2]]
  }
}
```

#### Visualization Example

```python
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

# Load transport flows
flows = gpd.read_file('transport_edges_with_flows_0.geojson')
flows = flows.to_crs(epsg=3857)  # Web Mercator for basemap

# Create map
fig, ax = plt.subplots(figsize=(12, 8))

# Plot flows with width proportional to volume
flows.plot(
    ax=ax,
    linewidth=flows['baseline_flow']/1000,
    color='red',
    alpha=0.7,
    label='Transport Flows'
)

# Add basemap
ctx.add_basemap(ax, crs=flows.crs, source=ctx.providers.OpenStreetMap.Mapnik)

plt.title('Transport Network Utilization')
plt.legend()
plt.tight_layout()
plt.show()
```

## Network Data

### Supply Chain Network (`sc_network_edgelist.csv`)

Complete supply chain relationships.

```csv
supplier_id,buyer_id,supplier_type,buyer_type,product,weight,baseline_flow
1001,1003,firm,firm,AGR_KHM,0.25,150000
1002,1003,firm,firm,AGR_KHM,0.35,200000
CHN,1003,country,firm,MAN_CHN,0.40,500000
1003,hh_1,firm,household,MAN_KHM,0.15,75000
```

#### Columns

| Column | Description |
|--------|-------------|
| `supplier_id` | Supplier identifier |
| `buyer_id` | Buyer identifier |
| `supplier_type` | Agent type (firm/country/household) |
| `buyer_type` | Agent type |
| `product` | Product/sector traded |
| `weight` | Relationship strength |
| `baseline_flow` | Economic flow value |

### Input-Output Table (`io_table.csv`)

Realized input-output flows.

```csv
,AGR_KHM,MAN_KHM,SER_KHM,HH_KHM,Export_CHN
AGR_KHM,50000,150000,25000,500000,100000
MAN_KHM,75000,200000,150000,800000,150000
SER_KHM,25000,100000,300000,600000,50000
```

## Analysis Results

### Loss Summary (`loss_summary.csv`)

Aggregate impact metrics.

```csv
metric,value,unit,description
total_production_loss,250000000,USD,Cumulative production loss
peak_production_loss,15000000,USD,Maximum single-period loss
affected_firms,1250,count,Firms with production loss > 0
total_consumption_loss,50000000,USD,Unmet household demand
welfare_loss,75000000,USD,Consumer welfare impact
recovery_time,45,days,Time to 95% recovery
```

### Sensitivity Analysis (`sensitivity_*.csv`)

Parameter sensitivity results from `disruption-sensitivity` simulations.

#### Structure
```csv
combination_id,io_cutoff,utilization,inventory_duration_targets.values.transport,household_loss,country_loss
0,0.01,0.8,1,1250000.5,890000.2
1,0.01,0.8,3,1180000.1,850000.8
2,0.01,0.8,5,1120000.9,820000.4
3,0.01,1.0,1,1180000.3,850000.1
4,0.01,1.0,3,1100000.7,810000.5
5,0.01,1.0,5,1050000.2,780000.3
6,0.1,0.8,1,980000.1,720000.8
7,0.1,0.8,3,920000.5,690000.2
8,0.1,0.8,5,890000.3,670000.1
9,0.1,1.0,1,950000.8,700000.4
10,0.1,1.0,3,900000.2,680000.7
11,0.1,1.0,5,870000.1,660000.2
```

#### Columns
- **`combination_id`** - Sequential identifier for parameter combination
- **Parameter columns** - One column per sensitivity parameter (dynamic)
- **`household_loss`** - Final household economic loss (model currency)
- **`country_loss`** - Final country trade loss (model currency)

#### Analysis Example
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load sensitivity results
df = pd.read_csv('sensitivity_20240101_120000.csv')

# Analyze io_cutoff impact
cutoff_impact = df.groupby('io_cutoff')['household_loss'].mean()
print("Average household loss by IO cutoff:")
print(cutoff_impact)

# Plot parameter sensitivity
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# IO cutoff sensitivity
df.boxplot(column='household_loss', by='io_cutoff', ax=axes[0])
axes[0].set_title('Household Loss vs IO Cutoff')

# Utilization sensitivity  
df.boxplot(column='household_loss', by='utilization', ax=axes[1])
axes[1].set_title('Household Loss vs Utilization')

plt.tight_layout()
plt.show()
```

### Regional Impacts (`loss_per_region_sector_time.csv`)

Detailed spatio-temporal analysis.

```csv
time,region,sector,production_loss,consumption_loss,employment_impact
0,Phnom_Penh,AGR_KHM,0,0,0
1,Phnom_Penh,AGR_KHM,500000,100000,50
2,Phnom_Penh,AGR_KHM,750000,150000,75
1,Siem_Reap,AGR_KHM,200000,50000,25
```

### Country Impacts (`loss_per_country.csv`)

International trade effects.

```csv
country,export_loss,import_disruption,trade_diversion,total_impact
CHN,25000000,15000000,5000000,45000000
THA,10000000,8000000,2000000,20000000
VNM,5000000,3000000,1000000,9000000
```

## Metadata Files

### Parameters (`parameters.yaml`)

Complete configuration snapshot.

```yaml
# Simulation configuration
simulation_type: "disruption"
scope: "Cambodia"
timestamp: "20241201_143022"

# Economic parameters
io_cutoff: 0.01
monetary_units_in_model: "mUSD"

# Events applied
events:
  - type: "transport_disruption"
    start_time: 10
    duration: 20
    
# Performance metrics
runtime_seconds: 3450
memory_peak_gb: 8.2
```

### Execution Log (`exp.log`)

Detailed execution information.

```
2024-12-01 14:30:22 INFO    Starting DisruptSC simulation for Cambodia
2024-12-01 14:30:25 INFO    Loading transport network... 
2024-12-01 14:31:15 INFO    Transport network loaded: 15,234 edges, 8,567 nodes
2024-12-01 14:31:20 INFO    Creating firms...
2024-12-01 14:32:45 INFO    Created 2,456 firms across 15 sectors
2024-12-01 14:35:12 INFO    Supply chain network created: 45,678 links
2024-12-01 14:42:30 INFO    Starting simulation...
2024-12-01 14:42:31 INFO    Time step 0: baseline state established
2024-12-01 14:42:45 INFO    Time step 10: applying transport disruption
2024-12-01 14:43:12 WARNING Transport disruption affecting 125 edges
2024-12-01 14:58:45 INFO    Simulation completed in 26m 23s
```

## Data Analysis Workflows

### Quick Impact Assessment

```python
import pandas as pd
import numpy as np

# Load key results
loss_summary = pd.read_csv('loss_summary.csv', index_col='metric')
regional_losses = pd.read_csv('loss_per_region_sector_time.csv')

# Key metrics
total_loss = loss_summary.loc['total_production_loss', 'value']
peak_loss = loss_summary.loc['peak_production_loss', 'value']
recovery_time = loss_summary.loc['recovery_time', 'value']

print(f"Total economic impact: ${total_loss:,.0f}")
print(f"Peak daily loss: ${peak_loss:,.0f}")
print(f"Recovery time: {recovery_time} days")

# Regional breakdown
regional_totals = regional_losses.groupby('region')['production_loss'].sum().sort_values(ascending=False)
print("\nMost affected regions:")
print(regional_totals.head())
```

### Sector Analysis

```python
# Sector vulnerability analysis
sector_impacts = regional_losses.groupby(['sector', 'time'])['production_loss'].sum().unstack(level=0)

# Plot sector impacts over time
sector_impacts.plot(figsize=(12, 6), title='Production Loss by Sector Over Time')
plt.ylabel('Production Loss')
plt.xlabel('Time Step')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Identify most vulnerable sectors
vulnerability = sector_impacts.max().sort_values(ascending=False)
print("Most vulnerable sectors:")
print(vulnerability.head())
```

### Spatial Analysis

```python
import geopandas as gpd
from shapely.geometry import Point

# Load spatial results
firms = gpd.read_file('firm_table.geojson')

# Calculate loss rates
firms['loss_rate'] = firms['production_loss'] / firms['eq_production']

# Create map of impacts
fig, ax = plt.subplots(figsize=(10, 8))
firms.plot(
    column='loss_rate',
    ax=ax,
    cmap='Reds',
    legend=True,
    markersize=firms['importance']*100,
    alpha=0.7
)
plt.title('Production Loss Rate by Firm')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Hotspot analysis
high_impact = firms[firms['loss_rate'] > 0.2]  # >20% loss
print(f"High-impact firms: {len(high_impact)}")
print(f"Average impact: {high_impact['loss_rate'].mean():.1%}")
```

### Network Analysis

```python
import networkx as nx

# Load supply chain network
sc_network = pd.read_csv('sc_network_edgelist.csv')

# Create network graph
G = nx.from_pandas_edgelist(
    sc_network, 
    source='supplier_id', 
    target='buyer_id',
    edge_attr='weight'
)

# Network metrics
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.3f}")

# Centrality analysis
centrality = nx.betweenness_centrality(G)
top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

print("\nMost central agents:")
for agent, score in top_central:
    print(f"{agent}: {score:.3f}")
```

## Performance and Optimization

### File Size Management

Large output files can impact performance:

```python
import os

def check_file_sizes(output_dir):
    """Check output file sizes."""
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            filepath = os.path.join(root, file)
            size_mb = os.path.getsize(filepath) / (1024*1024)
            if size_mb > 100:  # Files > 100MB
                print(f"Large file: {file} ({size_mb:.1f} MB)")

check_file_sizes('output/Cambodia/latest/')
```

### Memory-Efficient Loading

For large datasets:

```python
# Load data in chunks
def load_large_csv(filepath, chunksize=10000):
    """Load large CSV files in chunks."""
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        # Process chunk
        processed = chunk.groupby('region').sum()
        chunks.append(processed)
    return pd.concat(chunks)

# Use generators for JSON
def load_time_series(filepath):
    """Generator for time series data."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    for variable, time_series in data.items():
        yield variable, pd.DataFrame(time_series)
```

## Best Practices

### Data Management

1. **Archive results** - Copy important runs to permanent storage
2. **Document analysis** - Save analysis scripts with results
3. **Version tracking** - Record model version and parameters
4. **Compression** - Compress large output directories

### Reproducibility

1. **Save parameters** - Always archive the `parameters.yaml` file
2. **Record environment** - Document software versions
3. **Analysis scripts** - Version control analysis code
4. **Data lineage** - Track input data sources and versions

### Visualization

1. **Consistent scales** - Use same scales for comparison
2. **Color schemes** - Choose appropriate colormaps
3. **Interactive maps** - Use Folium/Plotly for web maps
4. **Export formats** - Save plots in vector formats (SVG/PDF)

### Performance

1. **Selective loading** - Only load needed variables
2. **Spatial indexing** - Use spatial indices for large datasets
3. **Parallel processing** - Use multiprocessing for analysis
4. **Memory monitoring** - Track memory usage during analysis