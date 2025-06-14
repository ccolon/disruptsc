# DisruptSupplyChain Model

## Data Repository Setup

**Important**: This repository contains only the model code. Input data is managed separately for security and size reasons.

### For New Users

**Step 1: Choose Data Setup Method**

Choose one of the following options to provide input data to the model:

**Option A: Git Submodule (Recommended for collaborators)**
```bash
# If you have access to the private disrupt-sc-data repository
git submodule add <disrupt-sc-data-repository-url> data
git submodule update --init
```

**Option B: Environment Variable (Flexible for different data locations)**
```bash
# Point to your local data directory
export DISRUPT_SC_DATA_PATH=/path/to/your/data/folder
```

**Option C: Local Input Folder (Simple for standalone use)**
```bash
# Create input folder and organize data by region
mkdir input
# Copy your data files following the structure described below
```

### Data Structure

Input data should be organized by region:
```
data/<region>/          # or input/<region>/ for local setup
├── Economic/           # MRIO tables, sector definitions
├── Transport/          # Infrastructure GeoJSON files
└── Spatial/           # Geographic disaggregation data
```

### Data Path Priority

The model automatically detects data location in this order:
1. `DISRUPT_SC_DATA_PATH` environment variable (highest priority)
2. `data/` folder (git submodule)
3. `input/` folder (local fallback)

### Large File Handling

Some input files exceed 50MB and may trigger GitHub warnings. Consider using Git Large File Storage (Git LFS) if you frequently update large data files.

**Step 2: Request Data Access**

Contact the maintainers for access to the input data repository or sample datasets.

## Installation

Git clone the project

You need python. The required libraries are in the "dsc-environment.yml" file. To set the right environment, 
we recommend using an "Anaconda" or "Miniconda" distribution. Then you can create the adequate environment by doing:
	
	 conda env create -f dsc-environment.yml

Note that I observed that, in some situations, this command gets stuck in the "solving environment" step.
I think we need to define the package versions in the yml. To be updated...

### Calling the script

Activate the right environment:

	conda activate dsc

The model is launched by the following command:

	python disruptsc/main.py <region> <optional argument>

Argument `region` corresponds to the region that is studied, for instance, `Tanzania`. This string should
be the same as:
- the data subfolder containing the region's data, e.g., "data/Tanzania" (or "input/Tanzania" if using local setup).
- the suffix of the user-defined parameters yaml file in the `parameter` folder, 
e.g., "parameter/user_defined_Tanzania.yaml".
This file should be created manually by the user.

The optional argument is made to allow for some part of the initial state to be resued in a subsequent call:
- `same_transport_network_new_agents`: the transport network is reused, but new agents, supplier-buyer links, 
logistic routes are generated
- `same_agents_new_sc_network`: the transport network and the agents are reused, but new supplier-buyer links, 
logistic routes are generated
- `same_sc_network_new_logistic_routes`: the transport network, agents, supplier-buyer links, are reused, 
but new logistic routes are generated
- `same_logistic_routes`: everything is reused
The files are cached in the `tmp` folder as pickle files. Note that if you change region, you will first need 
to call the script without optional argument.

The default parameters are defined in the "parameter/default.yaml" file. This file also defines the filepath 
to the input files. To change a parameter of filepath value, write it into the "parameter/user_defined_<region>.yaml"
file.


## Concepts

### Geographic structure

The model focuses on a country or group of countries. A country is divided into some administrative units 
(e.g., regions, districts, communes, cantones, etc.) We need to pick a relevant administrative level 
for which the economic data are compiled. We call it "regions" in the remainder of the text.

Regions are labelled using the country-relevant classification, for instance, the string `XXXXXX` where X are digits. 
We call this id the region code. It should always be a string, even if it consists of digits only, 
e.g., `123456` as string, not as integer.

The transport network is composed of edges and nodes. Edges are *LineString*, Nodes are *Points*. 
Edges are identified by an integer ID, so are nodes.

Each region is associated with one node in the transport network.


### Sector structure

We use one sector classification, such as ISIC Rev. 4, which depends on the data available on firms and on the 
input-output table available, and on their granularity. Each sector is identified by a trigram, 
such as `MTE` for Manufacture of Textiles. Note that imports are labeled by `IMP`.

### Objects

The main objects are the economic agents. There are three classes of agents:
- firms
- households
- countries

Firms, households, and countries are associated with nodes in the transport network. 
There is at most one firm per sector per region, one household per sector per region. 
Countries are associated to nodes which are located outside the country.

### Implementation choices

- For geographical data file, we use GeoJSON and not shapefiles. They should not be projected and 
use the *epsg:4326* coordinate reference system.


### Different versions of the model with different inputs needs

The model can be used in different version, depending on how supply-chain linkages are inputed.
Such versions are controlled by the `firm_data_type` parameter, which can takes three values:
- *disaggregating IO* uses a global input-table which gets disaggregated using regional economic data
- *mrio* use a multi-regional input-output table
- *supplier buyer network* uses supplier buyer network data [not yet implemented]


## Inputs

### Transport inputs are needed whatever the versions

Note that the filepath of each data files are defined in `parameter/filepaths_default.py`.
These default filepaths can be overriden using the `parameter/filepaths.py` path.

Within a subfolder `Transport`, there is a series of GeoJSON files and one YAML file.

#### Transport network files (GeoJSON)

There should be two GeoJSON files per transport mode, one for nodes and one for edges. 
Acceptable transport modes are : 'roads', 'airways', 'maritime', 'waterways', 'pipelines', 'multimodal'. 
For instance, for roads:
- `roads_edges.geojson`

If you use multiple modes, you need a `multimodal_edges.geojson` file.

The only accepted geometry is *LineString*. No *MultiLineString* is allowed.

Edges should contain at least the following attributes:
- `id`: int, one unique id per mode
- `end1`: int, the id of the node of the same mode where the linestring starts (leave empty for multimodal edges)
- `end2`: int, the id of the node of the same mode where the linestring ends (leave empty for multimodal edges)
- `surface`: paved or unpaved. If unknown you can put everything as "paved"
- `class`: class category, for instance primary, secondary, etc. If unknown you can leave it empty.
- `km`: length in km
- `multimodes` (for multimodal edges only): define which type of multimodel link, 
- for instance "roads-waterways" or "railways-maritime"
- `capacity`: maximum handling capacity per time step. If unknown you can leave it empty.
- `disruption`: flag here the disruption scenarios that affect this edge. Leave empty if unused.

They should not contain the following attributes:
- `index`

Note for multimodal edges:
- in multimodes, it should start with `mode1-mode2`. The order of the modes does not matter. 
- can write something after, e.g., 'roads-maritime-dom', but not before, e.g., 'roads-dom-maritime'.

Based on these input files, the model creates one *networkx.Graph* object representing the transport network.


### Inputs for *disaggregating IO* version

Three subfolders are needed:
- National: country-wide data derived from input--output tables or other country-wide data
- Subnational: subnational data, typically, for each region, population, location of the main city, sector size
- Trade: import, export, transit flows


#### National - Sector Table

A CSV file `sector_table.csv`. One row = one sector. Required columns:
- `sector`: the sector's trigram, for instance `AGR`
- `type`: one of 'agriculture', 'mining', 'manufacturing', utility', 'transport', 'trade', 'service'
- `output`: the total yearly output, derived from the input-output table. 
The unit should be the same as defined in the parameter `monetary_unit_in_model`.
- `final_demand`: the total yearly final demand, derived from the input-output table. 
The unit should be the same as defined in the parameter `monetary_unit_in_model`.
- `usd_per_ton`: the average monetary value, in USD, of a ton of good. 
This value can be computed from UN COMTRADE data, in which trade flows are both reported in tons and in USD. 
Set to 0 for sectors whose type is 'utility', 'transport', 'trade', 'service'
- `share_exporting_firms`: the percentage of the firms that export per sector. 
This value can be derived from country-specific data. 
Without good data, we can simply use for instance the share of export per sector.
- `supply_data`: the attribute of the 'region_data' file 
that the model will use to disaggregate the sectoral data
- `cutoff`: the cutoff value to apply. 
We will not model any firm for this sector in regions whose supply_data is below the cutoff

Note that, when no network data are used, the model will create firms based on the geospatial economic data. To speed up computation, firms that would be too small are dropped.

Example:

sector | type | output | final_demand | usd_per_ton | share_exporting_firms | supply_data | cutoff
--- | --- | --- | ---  | ---  | --- |-------------| --- 
AGR | agriculture | 415641365| 246213152 | 950 | 0.16 | ag_prod | 3.50E+06
... | ... | ... | ... | ... | ... | ...         | ... 


#### National - Technical coefficients

Technical coefficients are derived from the symmetric industry-by-industry input-output table. 
The matrix of technical coefficients are sometimes directly available. 
If only the supply and use tables are available, additional steps are required, which are not described here. 
Make sure that the sectors in rows and columns are the one to be used in the model, i.e., 
that there are consistent with the firm-level data.

For the model, a CSV file `tech_coef_matrix.csv` should be provided with the following structure. 

Example:

|  | AGI | FOR | ...
--- | --- | --- | --- 
**AGI** | 0.11 | 0 | ...
**FOR** | 0.01 | 0.2 | ...
... | ... | ... | ...



#### National - Inventory Duration Target

Firms hold inventory of inputs, quantified by the number of weeks the firm can continue producing 
without new supply, called "inventory duration target". Those targets are defined nationally 
for each combination of input\*sector.

Example:

input_sector | buying_sector | inventory_duration_target
--- | --- | --- 
TRD | AGR | 3.5
... | ... | ...


#### Subnational - Region Data

In this file are summarized the socioeconomic and spatial data on economic production for each region. 
A GeoJSON file is expected.
- `admin_code`: a string with the administrative code of the region
- `geometry`: a 'Point', which represent where the firms and household modelled for this region will be located. 
Typically the location of the largest city in the region, or, as a second best option, the region's centroid.
- `population`: the number of people in the region
- economic supply data: at least one column per sector which captures the size of the sector in this region. 
For instance, there could be the number of employees in the manufacturing of basic metals, 
the total sales of construction, and the value of production of agricultural products. 
The column name should corresponds to the values of the `supply_data` column in the _sector table_

A GeoJSON file `economic_data.geojson` with Points and the following attribute table:

Example:

admin_code | nb_workers_MAN | nb_workers_ELE | crop_production | ... | geometry
--- | --- | --- | --- | --- | ---
0101 | 124 | 12 | 465120 | ... | Point(-45.15631, 0.48654)
... | ... | ... | ... | ... | ...


#### Trade - Country Transit Matrix

A CSV file representing a matrix. Country codes are row and column headers. It represents the yearly trade flow 
from the row country to the column country that goes through the country-under-study's transport network. 
The unit should be the same as defined in the parameter `monetary_unit_in_model`.

Example:

... | BDI | COD | ...
--- | --- | --- | --- 
**BDI** | 4563 | 4516 | ...
... | ... | ... | ...

#### Trade - Import Table

A CSV file representing a matrix. Country codes are row headers. Sector codes are column headers. 
It represents the yearly imports from the different countries, in row, to the domestic sectors, in column. 
The unit should be the same as defined in the parameter `monetary_unit_in_model`.

Example:

... | AGR | FOR | ...
--- | --- | --- | --- 
**BDI** | 132 | 0 | ...
... | ... | ... | ...


#### Trade - Export Table

A CSV file representing a matrix. Country codes are row headers. Sector codes are column headers. 
It represents the yearly exports from the domestic sectors, in columns, to the foreigh countries, in row. 
The unit should be the same as defined in the parameter `monetary_unit_in_model`.

Example:

... | AGR | FOR | ...
--- | --- | --- | --- 
**BDI** | 2 | 456 | ...
... | ... | ... | ...


### Inputs for the *mrio* version

One subfolder is needed: *Network*

#### MRIO (CSV)

A CSV file which contains the multi-regional input-output data.

It is a double-indexed matrix, with the following structure :

... | ...     | AFG | AFG | FRA | FRA | AFG | FRA | ROW
--- |---------| --- | --- | --- | --- | --- | --- | ---
... | ...     | Sector1 | Sector2 | Sector1 | Sector2 | final_demand | final_demand | exports
AFG | Sector1 | ... | ... | ... | ... | ... | ... | ...
AFG | Sector2 | ... | ... | ... | ... | ... | ... | ...
FRA | Sector1 | ... | ... | ... | ... | ... | ... | ...
FRA | Sector2 | ... | ... | ... | ... | ... | ... | ...
ROW | imports | ... | ... | ... | ... | ... | ... | ...

The first row/column indicates the "admin_code" of the regions.
The second row/column indicates the "sector_code" of the sectors.

#### region_table (GeoJSON)

A GeoJSON file with *Point* geometries indicating the origin-destination point associated to each region.
It should contain the following attributes:
- `admin_code`: should correspond to the admin_code of the mrio.csv
- `geometry`: Point

### sector_table (CSV)

A CSV file `sector_table.csv`. One row = one region*sector. Required columns:
- `sector`: the combination of <region_code>_<sector_code>
- `type`: one of 'agriculture', 'mining', 'manufacturing', utility', 'transport', 'trade', 'service'
- `output`: the total yearly output, derived from the input-output table. 
The unit should be the same as defined in the parameter `monetary_unit_in_model`.
- `final_demand`: the total yearly final demand, derived from the input-output table. 
The unit should be the same as defined in the parameter `monetary_unit_in_model`.
- `usd_per_ton`: the average monetary value, in USD, of a ton of good. 
This value can be computed from UN COMTRADE data, in which trade flows are both reported in tons and in USD. 
Set to 0 for sectors whose type is 'utility', 'transport', 'trade', 'service'
- `share_exporting_firms`: the percentage of the firms that export per sector. 
This value can be derived from country-specific data. 
Without good data, we can simply use for instance the share of export per sector.
- `supply_data`: the attribute of the 'region_data' file 
that the model will use to disaggregate the sectoral data
- `cutoff`: the cutoff value to apply. 
We will not model any firm for this sector in regions whose supply_data is below the cutoff

Note that, when no network data are used, the model will create firms based on the geospatial economic data. To speed up computation, firms that would be too small are dropped.

Example:

sector | type | output | final_demand | usd_per_ton | share_exporting_firms | supply_data | cutoff
--- | --- | --- | ---  | ---  | --- |-------------| --- 
AFG_agriculture | agriculture | 415641365| 246213152 | 950 | 0.16 | ag_prod | 3.50E+06
... | ... | ... | ... | ... | ... | ...         | ... 


#### Inventory Target (CSV)

Firms hold inventory of inputs, quantified by the number of weeks the firm can continue producing 
without new supply, called "inventory duration target". Those targets are defined nationally 
for each combination of input * country*sector.

Example:

input_sector | buying_sector | inventory_duration_target
--- | --- | --- 
ESP_Electrical and Machinery | ESP_Recycling | 13
ESP_Electrical and Machinery | SYR_Others | 14
ESP_Electrical and Machinery | TUR_Public Administration  | 10
ESP_Electrical and Machinery | USR_Agriculture  | 8
... | ... | ...


## Indicatives steps for data prep for *disaggregating IO* version 

*Requisite:*
- Say you have sales data per sector per region, *sales_csv*, preprated from business census data. 
It should be coded with a relevant sector classification.
- You should also have a list of countries, or group of countries, to represent the rest of the world. 
Typically, isolate the neighbooring countries, or those that very relevant in terms of trade, 
and then group the rest based on continents. Code the countries with a trigram. 
For instance, VNM for Vietnam, THA for Thailand, ASI for Asia, OCE for Oceania, AFR for Africa, EUR for Europe, AME for America.

*Steps:*

**1. Prepare an initial sector table**

- Prepare a *sector_csv*, with all the sector and their corresponding identifier 
(recommended identifier: a trigram, e.g. CER for cereal production or MTE for manufacturing of textile)
- The column with the identifier should be called *sector*

**2. Prepare the region data**

- Prepare a *region_geojson*, with all region, their adequate identifier 
(typically the administrative code, beware whether you want it to me string or integer), 
the point of their main city as geometry
- Make sure the coordinates system is EPSG 4326 (unprojected, degrees)
- The column with the admin code should be called *admin_code*
- Make sure that in your *sales_csv*, the regions and sectors are adequately coded
- Create the *region_data_geojson* (see above) from *region_geojson* and *sales_csv*, 
which includes population per region

**3. Add the sector cutoff to the sector table**

- Add a column *supply_data* in the sector table: 
it should correspond to the name of the attribute of the regional data (*region_geojson*) 
contains the economic data of that sector
- Add a column *cutoff* in the sector table: only region with data above this cutoff will be 
modeled as production place for that sector

**4. Prepare the technical coefficient matrix**

- Get the symmetric industry-by-industry input--output (IO) table with the adequate sector classification
	- Note that you may need the adjust the sector classification of the IO table yourself, 
  and/or the one of the sales data. What is important is that the same sector classification is used throughout the data
- Open it in excel (or equivalent) and control that you get the GDP using the 3 approaches 
(income, production, expenditure)
- Create the technical coefficient matrix, add the row for imports, coded IMP

**5. Add output, final demand, share_exporting_firms to the sector table**

- Use the input--output table of step 2 to retrieve total output, final demand, and % of export 
(exports / total output) per sector
- Add those three columns to the sector table with the following names: output, final_demand, share_exporting_firms
	- Note that we use here % of export as a proxy for the proportion of firms that exports. 
  If better data are available on this, use it instead of % of export

**6. Prepare the import and export tables**

- Get trade data for the relevant year, choose the country as "Partner", get the csv file, 
such as *comtrade_2016_EcuadorAsPartner_AG2.csv*
- Prepare a country mapping table, which maps the 3-digit ISO code of all countries listed 
in the ComTrade data to your country classification
- Prepare a sector mapping table, which maps the ComTrade commodity code 
(see https://unstats.un.org/unsd/classifications/Econ) into your sector classification
- Map the trade data into your country and sector classification
- Create the import table and the export table

**7. Prepare the transit matrix**

- Identify the pair of neighboring countries, whose physical trade flows could go through the country 
under study (e.g., Peru - Colombia for Ecuador)
- For each pair, get the total import and export of good between the two countries of the pair on 
ComTrade for the relevant year (not splitted by sector)
- With any data available, estimate how much of these trade flows goes through the country
- Produce the resulting transit matrix

**8. Add usd_per_ton to the sector table**

- Get comTrade data, with the country under study as reporter, splitted by commodity, for the relevant year 
(not split by partners)
- Check that the trade flows are both in qty and in USD
- Map it onto the sector classification
- Calculate the ratio USD per ton for each sector
- Add this column to the sector table

**9. Calculate the inventory target table**

- Get the inventory data and map ii into the sector classification
- Calculate the inventory duration targets for each pair of supplying sector to buying sector
- Create the inventory target table


## Running a disruption scenario

In the user_defined parameters file, set _simulation_type_ to `disruption`.
In the `events` parameter, write a list of disruption events. Each disruption event is a dictionary.

There are two types of events: transport disruptions and capital destructions.

A transport disruption is defined by:
- _type_ [str]: `transport_disruption`
- _description_type_ [str]: `edge_attributes` (more are possible, to be described)
- _attribute_ [str]: the attribute of the edge.geojson file that the model will use to identify the edge to disrupt, 
  for isntance
- _value_ [list[str]]: the value of the attribute that indicates wether that an edge is disrupted or not. 
  This is a list, so multiple values can be given
- _start_time_ [int]: the time step at which the disruption starts
- _duration_ [int]: the amount of time steps the disruption lasts. After that, the edge is back to normal
 
A capital destruction is defined by:
- _type_ [str]: `- type: `capital_destruction`
- _description_type_ [str]: `region_sector_file`
- `region_sector_filepath` [str]: filepath of the CSV file describing the amount of destroyed capital in each
  region sector, e.g., "Disruption/earthquake_capital_destruction.csv" (other formats to be described)
- `unit` [str]: monetary units used in the region_sector_filepath, possible values are USD, kUSD, mUSD 
- _reconstruction_market_ [bool]: if `True`, a reconstruction market is created, and capital is rebuilt. 
  The parameters of this market is, until now, hard-coded in the model.
  If `False`, the capital is never rebuilt.
- _start_time_ [int]: the time step at which the disruption starts


## Model run

The model is launched by the following command:

	python disruptsc/main.py <region> <optional argument>

See the section "Calling the script" for more details.

This is the general workflow of the model:

```mermaid
graph TD;
    A[1. Setup transport network]-->B[2. Setup agents];
    B-->C[3. Setup supply chain network];
    C-->D[4. Setup logistic routes];
    D-->E[5. Initialize economic variables];
    E-->F[6. Run time step 0 to set up flows];
    F-->G[7. Run disruption scenario];
```

Effect of the optional argument in the model run:
- When using no optional argument, the model starts at step 1.
- When using `same_transport_network_new_agents`, the model starts at step 2.
- When using `same_agents_new_sc_network`, the model starts at step 3.
- When using `same_sc_network_new_logistic_routes`, the model starts at step 4.
- When using `same_logistic_routes`, the model starts at step 5.

Note that only step 3, setup supply chain network, is stochastic. To capture the stochasticity,
the model can be run multiple times restarting at step 3.
