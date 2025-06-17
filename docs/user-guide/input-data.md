# Input Data

This comprehensive guide explains all input data files required by DisruptSC, their formats, and how to prepare them.

## Data Organization

All input data must be organized by scope (region/country):

```
data/<scope>/               # e.g., data/Cambodia/
├── Economic/               # Economic and sector data
├── Transport/              # Infrastructure networks  
├── Spatial/                # Geographic disaggregation
└── Disruption/            # Disruption scenarios (optional)
```

## Economic Data

### MRIO Table (`Economic/mrio.csv`)

The Multi-Regional Input-Output table is the core economic data file.

#### Structure

```csv
region_sector,AGR_REG1,MAN_REG1,SER_REG1,HH_REG1,Export_CHN,Import_CHN
AGR_REG1,150.5,75.2,25.0,500.0,100.0,0.0
MAN_REG1,50.0,200.0,150.0,800.0,150.0,0.0
SER_REG1,25.0,100.0,300.0,600.0,50.0,0.0
Import_CHN,10.0,50.0,20.0,100.0,0.0,0.0
```

#### Required Features

- **Square matrix** - Same sectors in rows and columns
- **Monetary units** - Consistent currency (USD, kUSD, mUSD)
- **Balanced flows** - Row sums ≈ column sums for each sector
- **Final demand** - Household columns (HH_RegionName)
- **Trade flows** - Export/Import columns for international trade

#### Region-Sector Naming

Use format: `{SectorCode}_{RegionCode}`

**Examples:**
- `AGR_KHM` - Agriculture in Cambodia
- `MAN_THA` - Manufacturing in Thailand
- `SER_VNM` - Services in Vietnam

#### Special Columns/Rows

| Type | Format | Purpose |
|------|--------|---------|
| **Households** | `HH_{Region}` | Final consumption demand |
| **Exports** | `Export_{Country}` | International exports |
| **Imports** | `Import_{Country}` | International imports |

#### Data Quality Requirements

```python
# Validation checks performed
import pandas as pd

mrio = pd.read_csv('mrio.csv', index_col=0)

# 1. Square matrix
assert mrio.shape[0] == mrio.shape[1], "MRIO must be square"

# 2. Non-negative values
assert (mrio >= 0).all().all(), "Negative values not allowed"

# 3. Reasonable magnitudes
assert mrio.max().max() < 1e12, "Values seem too large"

# 4. No empty rows/columns
assert not (mrio.sum(axis=1) == 0).any(), "Empty sectors found"
```

### Sector Table (`Economic/sector_table.csv`)

Defines characteristics for each region-sector combination.

#### Required Columns

```csv
sector,type,output,final_demand,usd_per_ton,share_exporting_firms,supply_data,cutoff
AGR_KHM,agriculture,2000000,500000,950,0.16,ag_prod,3500000
MAN_KHM,manufacturing,5000000,800000,2864,0.45,man_prod,5000000
SER_KHM,service,3000000,600000,0,0.10,ser_emp,2000000
```

#### Column Specifications

| Column | Type | Description | Units | Range |
|--------|------|-------------|-------|-------|
| `sector` | string | Region_sector identifier | - | Must match MRIO |
| `type` | string | Sector category | - | See types below |
| `output` | float | Total yearly output | Model currency | > 0 |
| `final_demand` | float | Total yearly final demand | Model currency | ≥ 0 |
| `usd_per_ton` | float | USD value per ton | USD/ton | ≥ 0 |
| `share_exporting_firms` | float | Export participation rate | fraction | 0-1 |
| `supply_data` | string | Spatial disaggregation attribute | - | Must exist in firms.geojson |
| `cutoff` | float | Minimum firm size threshold | Model currency | ≥ 0 |

#### Sector Types

| Type | Description | Transport | USD/ton |
|------|-------------|-----------|---------|
| `agriculture` | Farming, fishing, forestry | Physical goods | > 0 |
| `mining` | Extraction industries | Physical goods | > 0 |
| `manufacturing` | Processing industries | Physical goods | > 0 |
| `utility` | Electricity, water, waste | Services | 0 |
| `transport` | Transport services | Services | 0 |
| `trade` | Wholesale, retail | Services | 0 |
| `service` | All other services | Services | 0 |

!!! tip "USD per Ton"
    
    - Set to **> 0** for physical goods that are transported
    - Set to **0** for services that don't require physical transport
    - Use UN COMTRADE data for realistic values
    - Typical ranges: 500-5000 USD/ton for most goods

### Firm-Level Data (Network Mode Only)

Required only when using `firm_data_type: "supplier-buyer network"`.

#### Firm Table (`Economic/firm_table.csv`)

```csv
id,sector,region,output,employees,importance
1001,AGR_KHM,Phnom_Penh,500000,50,0.15
1002,AGR_KHM,Siem_Reap,750000,75,0.25
1003,MAN_KHM,Phnom_Penh,2000000,200,0.45
```

| Column | Description | Units |
|--------|-------------|-------|
| `id` | Unique firm identifier | integer |
| `sector` | Region_sector code | string |
| `region` | Sub-region location | string |
| `output` | Annual output | model currency |
| `employees` | Number of employees | count |
| `importance` | Relative importance weight | 0-1 |

#### Location Table (`Economic/location_table.csv`)

```csv
firm_id,long,lat,transport_node,admin_level1
1001,-104.532,40.123,node_456,Phnom_Penh
1002,-104.445,40.234,node_789,Siem_Reap
1003,-104.612,40.087,node_234,Phnom_Penh
```

#### Transaction Table (`Economic/transaction_table.csv`)

```csv
supplier_id,buyer_id,product_sector,transaction,is_essential
1001,1003,AGR_KHM,150000,true
1002,1003,AGR_KHM,200000,false
1004,1003,MAN_THA,500000,true
```

## Transport Networks

All transport files must be GeoJSON with **LineString** geometry.

### Required Files

#### Roads (`Transport/roads_edges.geojson`)

Primary transport network for domestic movements.

```json
{
  "type": "Feature",
  "properties": {
    "highway": "primary",
    "length_km": 15.2,
    "max_speed": 80,
    "capacity": 2000
  },
  "geometry": {
    "type": "LineString",
    "coordinates": [[-104.5, 40.1], [-104.4, 40.2]]
  }
}
```

**Required properties:**
- `highway` or `road_type` - Road classification
- `length_km` - Length in kilometers
- Optional: `max_speed`, `capacity`, `surface_type`

#### Maritime (`Transport/maritime_edges.geojson`)

International shipping and ferry routes.

```json
{
  "type": "Feature", 
  "properties": {
    "route_type": "shipping",
    "length_km": 450.0,
    "port_from": "Sihanoukville",
    "port_to": "Ho_Chi_Minh"
  },
  "geometry": {
    "type": "LineString",
    "coordinates": [[-103.5, 10.6], [-106.8, 10.8]]
  }
}
```

### Optional Transport Modes

#### Railways (`Transport/railways_edges.geojson`)
```json
{
  "properties": {
    "rail_type": "freight",
    "gauge": "standard",
    "length_km": 25.0
  }
}
```

#### Airways (`Transport/airways_edges.geojson`)
```json
{
  "properties": {
    "route_type": "cargo",
    "airport_from": "PNH", 
    "airport_to": "BKK",
    "length_km": 400.0
  }
}
```

#### Waterways (`Transport/waterways_edges.geojson`)
```json
{
  "properties": {
    "waterway": "river",
    "navigable": true,
    "length_km": 80.0
  }
}
```

#### Pipelines (`Transport/pipelines_edges.geojson`)
```json
{
  "properties": {
    "pipeline_type": "oil",
    "diameter_mm": 600,
    "length_km": 120.0
  }
}
```

#### Multimodal (`Transport/multimodal_edges.geojson`)

Connections between different transport modes.

```json
{
  "properties": {
    "connection_type": "port_to_road",
    "from_mode": "maritime",
    "to_mode": "roads",
    "transfer_time": 2.0
  }
}
```

### Transport Network Requirements

1. **Connectivity** - Network must be connected for routing
2. **Coordinate system** - Use WGS84 (EPSG:4326)
3. **Valid geometry** - No self-intersections or invalid coordinates
4. **Realistic distances** - Length should match geometric distance

## Spatial Data

Geographic disaggregation files with **Point** geometry.

### Households (`Spatial/households.geojson`)

Population distribution for household placement.

```json
{
  "type": "Feature",
  "properties": {
    "region": "Phnom_Penh",
    "population": 15000,
    "admin_level1": "Phnom_Penh",
    "admin_level2": "Khan_Chamkar_Mon"
  },
  "geometry": {
    "type": "Point", 
    "coordinates": [-104.866, 11.555]
  }
}
```

**Required properties:**
- `region` - Region identifier (must match MRIO regions)
- `population` - Population count (optional, defaults to 1)

### Countries (`Spatial/countries.geojson`)

International trade entry/exit points.

```json
{
  "type": "Feature",
  "properties": {
    "region": "CHN",
    "entry_type": "border_crossing",
    "capacity": 1000
  },
  "geometry": {
    "type": "Point",
    "coordinates": [-104.0, 12.0]
  }
}
```

**Required properties:**
- `region` - Country identifier (must match MRIO import/export countries)

### Firms (`Spatial/firms.geojson`)

Economic activity distribution for firm placement.

```json
{
  "type": "Feature",
  "properties": {
    "region": "Phnom_Penh",
    "ag_prod": 150,
    "man_prod": 85,  
    "ser_emp": 200,
    "admin_level1": "Phnom_Penh"
  },
  "geometry": {
    "type": "Point",
    "coordinates": [-104.866, 11.555]
  }
}
```

**Required properties:**
- `region` - Region identifier (must match MRIO regions)
- Sector-specific attributes referenced in `sector_table.csv` `supply_data` column

**Sector attribute examples:**
- `ag_prod` - Agricultural production value
- `man_prod` - Manufacturing output
- `min_prod` - Mining production  
- `ser_emp` - Service employment
- `pop_density` - Population density

## Disruption Data (Optional)

### Capital Destruction (`Disruption/capital_destruction.csv`)

Economic damage from disasters.

```csv
region_sector,capital_destroyed,unit
AGR_KHM,50000000,USD
MAN_KHM,120000000,USD
SER_KHM,30000000,USD
```

**Columns:**
- `region_sector` - Affected sector
- `capital_destroyed` - Damage amount
- `unit` - Currency unit

### Transport Disruptions

Specify transport disruptions using edge attributes:

```yaml
# In parameter file
events:
  - type: "transport_disruption"
    description_type: "edge_attributes"
    attribute: "highway"
    value: ["primary", "trunk"]
    start_time: 10
    duration: 20
```

## Data Preparation Guidelines

### Data Sources

#### Economic Data
- **National accounts** - Statistical offices
- **Input-output tables** - OECD, national statistics
- **GTAP database** - Global trade data
- **WIOD/EORA** - Multi-regional databases

#### Transport Networks
- **OpenStreetMap** - Road networks
- **Natural Earth** - Maritime routes
- **National datasets** - Railway, pipeline networks
- **Port authorities** - Maritime connections

#### Spatial Data
- **Census data** - Population distribution
- **Economic surveys** - Firm locations
- **Satellite data** - Land use patterns
- **Administrative boundaries** - Regional definitions

### Data Processing

#### Coordinate Systems
```python
import geopandas as gpd

# Ensure WGS84 projection
gdf = gpd.read_file('transport_data.geojson')
gdf = gdf.to_crs('EPSG:4326')
gdf.to_file('transport_edges.geojson', driver='GeoJSON')
```

#### Unit Conversion
```python
# Convert currency units
mrio_usd = mrio_original * 1e6  # Convert mUSD to USD

# Convert distance units  
length_km = length_miles * 1.60934
```

#### Data Validation
```python
# Check for missing values
assert not data.isnull().any().any(), "Missing values found"

# Verify coordinate bounds
assert (gdf.bounds.minx >= -180).all(), "Invalid longitude"
assert (gdf.bounds.maxy <= 90).all(), "Invalid latitude"

# Check data types
assert pd.api.types.is_numeric_dtype(data['output']), "Output must be numeric"
```

## Quality Assurance

### Automated Validation

Run comprehensive validation before simulations:

```bash
python validate_inputs.py Cambodia --comprehensive
```

### Manual Checks

#### Economic Consistency
- MRIO balance: row sums ≈ column sums
- Sector totals match national accounts
- Reasonable sector shares and ratios

#### Spatial Consistency  
- Points within country boundaries
- Transport networks connected
- Realistic distances and travel times

#### Data Completeness
- All required files present
- No missing values in critical fields
- Consistent identifiers across files

### Common Data Issues

!!! failure "MRIO Imbalance"
    
    **Problem:** Row sums ≠ column sums
    **Solution:** Check aggregation, add balancing items
    
    ```python
    # Check balance
    row_sums = mrio.sum(axis=1)
    col_sums = mrio.sum(axis=0) 
    balance = (row_sums - col_sums) / row_sums
    print(f"Max imbalance: {balance.abs().max():.1%}")
    ```

!!! failure "Disconnected Networks"
    
    **Problem:** Transport network has isolated components
    **Solution:** Add connecting edges, check topology
    
    ```python
    import networkx as nx
    G = nx.from_pandas_edgelist(edges, source='from', target='to')
    components = list(nx.connected_components(G))
    print(f"Network has {len(components)} components")
    ```

!!! failure "Coordinate Issues"
    
    **Problem:** Invalid or swapped coordinates
    **Solution:** Verify coordinate order (lon, lat)
    
    ```python
    # Check coordinate bounds
    print(f"Longitude range: {gdf.bounds.minx.min():.2f} to {gdf.bounds.maxx.max():.2f}")
    print(f"Latitude range: {gdf.bounds.miny.min():.2f} to {gdf.bounds.maxy.max():.2f}")
    ```

## Best Practices

### Data Management

1. **Version control** - Track data changes with git LFS
2. **Documentation** - Record data sources and processing steps
3. **Backup** - Maintain copies of original raw data
4. **Validation** - Always validate before production runs

### Performance Optimization

1. **File sizes** - Keep GeoJSON files under 50MB when possible
2. **Precision** - Round coordinates to appropriate precision (5-6 decimal places)
3. **Compression** - Use gzip for large CSV files
4. **Indexing** - Ensure proper indexing for database files

### Collaboration

1. **Standards** - Follow consistent naming conventions
2. **Metadata** - Include data dictionaries and README files
3. **Access** - Use appropriate data sharing mechanisms
4. **Privacy** - Ensure compliance with data protection regulations