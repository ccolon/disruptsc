# DisruptSupplyChain Model

## Concepts

### Geographic structure

The model focuses on a country. A country is divided into some administrative units (e.g., regions, districts, communes, cantones, etc.) We need to pick a relevant administrative level for which the economic data are compiled. We call it "districts" in the remainder of the text.

Districts are labelled using the country-relevant classification, for instance, the string `XXXXXX` where X are digits. We call this id the district code. It should always be a string, even if it consists of digits only, e.g., `123456` as string, not as integer.

The transport network is composed of edges and nodes. Edges are *LineString*, Nodes are *Points*. Edges are identified by an integer ID, so are nodes.

Each district is associated with one node in the transport network.

### Sector structure

We use one sector classification, such as ISIC Rev. 4, which depends on the data available on firms and on the input-output table available, and on their granularity. Each sector is identified by a trigram, such as `MTE` for Manufacture of Textiles. Note that imports are labeled by `IMP`.

### Objects

The main objects are the economic agents. There are three classes of agents:
- firms
- households
- countries

Firms, households, and countries are associated with nodes in the transport network. There is at most one firm per sector per district, one household per sector per district. Countries are associated to nodes which are located outside of the country.

### Implementation choices

- For geographical data file, we use GeoJSON and not shapefiles. They should all be unprojected, using *epsg:4326*.


## Using the model

### Calling the script

The model is launched by the following command `python code\mainBase.py <optional argument>`

The optional argument is made to allow for some part of the initial state to be resued in a subsequent call:
- `same_transport_network_new_agents`: the transport network is reused, but new agents, supplier-buyer links, logistic routes are generated
- `same_agents_new_sc_network`: the transport network and the agents are reused, but new supplier-buyer links, logistic routes are generated
- `same_sc_network_new_logistic_routes`: the transport network, agents, supplier-buyer links, are reused, but new logistic routes are generated
- `same_logistic_routes`: everything is reused


## Inputs

Create a subdirectory in the `input` directory, whose name should correspond to the `input_folder` variable given in the `parameters.py` file. Usually, this name is the country under study. Within this subdirectory, create 5 subdirectories:
- Disruption: specific list of transport nodes or edges to test (optional)
- National: country-wide data derived from input--output tables or other country-wide data
- Subnational: subnational data, typically, for each district, population, location of the main city, sector size
- Trade: import, export, transit flows
- Transport: the transport network

Note that the filepath of each data files are defined in `parameter/filepaths_default.py`. These default filepaths can be overriden using the `parameter/filepaths.py` path.


### Transport

#### Transport network files

There should be two GeoJSON files per transport mode, one for nodes and one for edges. Acceptable transport modes are : 'roads', 'airways', 'maritime', 'waterways', 'multimodal'. For instance, for roads:
- `roads_nodes.geojson`
- `roads_edges.geojson`
There is only one file for the multimodal layer, which describe the edges. There is no multimodal node.

The edge's geometry is *LineString*, the node's geometry is *Point*.

Nodes should contains at least the following attributes:
- `id`: int, one unique id per mode

Edges should contains at least the following attributes:
- `id`: int, one unique id per mode
- `surface`: paved or unpaved. If unknown you can put everything as "paved"
- `class`: class category, for instance primary, secondary, etc. If unknown you can leave it empty.
- `km`: length in km
- `multimodes` (for multimodal edges only): define which type of multimodel link, for instance "roads-waterways" or "railways-maritime"
- `capacity`: maximum handling capacity per time step. If unknown you can leave it empty.

Nodes and Edges should not contain the following attributes:
- `index`

Based on these input files, the model creates one *networkx.Graph* object representing the transport network.

Multimodal edges:
- in multimodes, it should start with `mode1-mode2`. The order of the modes does not matter. We can write something after, e.g., 'roads-maritime-dom', but not before, e.g., 'roads-dom-maritime'.


#### Transport Parameters

A yaml file `transport_parameters.yaml` with the following structure. It needs to be adjusted to the transport modes modelled.

	speeds: #km/hour
	  roads
	    paved: 31.4
	    unpaved: 15
	  railways: 23
	  waterways: 7
	  maritime: 35

	loading_time: #hours 
	  roads-waterways: 5
	  roads-maritime: 12
	  roads-railways: 12
	  railways-maritime: 24
	  waterways-maritime: 24

	variability: #as fraction of travel time
	  roads:
	    paved: 0.01 
	    unpaved: 0.075
	  railways: 0.02
	  waterways: 0.02
	  maritime: 0.005
	  multimodal:
	    roads-waterways: 0.1
	    roads-maritime: 0.1
	    roads-railways: 0.1
	    railways-maritime: 0.1
	    waterways-maritime: 0.1

	transport_cost_per_tonkm: #USD/(ton*km)
	  roads:
	    paved: 0.053
	    unpaved: 0.1
	  railways: 0.04
	  waterways: 0.029
	  maritime: 0.0017

	loading_cost_per_ton: #USD/ton
	  roads-waterways: 2.2
	  roads-maritime-shv: 2.2
	  roads-maritime-vnm: 2.2
	  roads-railways: 5
	  railways-maritime: 2.2
	  waterways-maritime: 2.2

	custom_cost: #USD/ton
	  roads: 27
	  waterways: 27
	  maritime: 27

	custom_time: #hours
	  roads: 1.5
	  waterways: 6
	  maritime: 2
	  multimodal: 2

	travel_cost_of_time: 0.49 #USD/hour

	variability_coef: 0.44 #USD/hour


#### Contraints on transport modes (optional)

An additional file `transport_modes.csv` can be used to prevent specific supply-chains flows from taking specific transport modes. 
To be further described


### National


#### Sector Table

A CSV file `sector_table.csv`. One row = one sector. Required columns:
- `sector`: the sector's trigram, for instance `AGR`
- `type`: one of 'agriculture', 'mining', 'manufacturing', utility', 'transport', 'trade', 'services'
- `output`: the total yearly output, derived from the input-output table. The unit should be the same as defined in the parameter `monetary_unit_in_model`.
- `final_demand`: the total yearly final demand, derived from the input-output table. The unit should be the same as defined in the parameter `monetary_unit_in_model`.
- `usd_per_ton`: the average monetary value, in USD, of a ton of good. This value can be computed from UN COMTRADE data, in which trade flows are both reported in tons and in USD. Set to 0 for sectors whose type is 'utility', 'transport', 'trade', 'services'
- `share_exporting_firms`: the percentage of the firms that export per sector. This value can be derived from country-specific data. Without good data, we can simply use for instance the share of export per sector.

Example:

sector | type | output | final_demand | usd_per_ton | share_exporting_firms
--- | --- | --- | ---  | ---  | --- 
AGR | agriculture | 415641365| 246213152 | 950 | 0.16
... | ... | ... | ... | ... | ... 



#### Technical coefficients

Technical coefficients are derived from the symmetric industry-by-industry input-output table. The matrix of technical coefficients are sometimes directly available. If only the supply and use tables are available, additional steps are required, which are not described here. Make sure that the sectors in rows and columns are the one to be used in the model, i.e., that there are consistent with the firm-level data.

For the model, a CSV file `tech_coef_matrix.csv` should be provided with the following structure. 

Example:

|  | AGI | FOR | ...
--- | --- | --- | --- 
**AGI** | 0.11 | 0 | ...
**FOR** | 0.01 | 0.2 | ...
... | ... | ... | ...




#### Sector firm cutoff

The model will create firms based on the geospatial economic data. To speed up computation, firms that would be too small are dropped. The CSV files `sector_firm_cutoffs.csv` define the cutoff to apply for each sector.
- `sector`: the trigram of the sector
- `supply_data`: the attribute of the 'district_data' file that the model will use to disaggregate the sectoral data
- `cutoff`: the cutoff value to apply. We will not model any firm for this sector in districts whose supply_data is below the cutoff

Example:

sector | supply_data | cutoff
--- | --- | --- | --- 
CRO | ag_prod | 3.50E+06
MIN | nb_workers_MIN | 200
MFO | sales_MFO | 200
... | ... | ... | ...


#### Inventory Duration Target

Firms hold inventory of inputs, quantified by the number of weeks the firm can continue producing without new supply, called "inventory duration target". Those targets are defined nationally for each combination of input\*sector.

Example:

input_sector | buying_sector | inventory_duration_target
--- | --- | --- 
TRD | AGR | 3.5
... | ... | ...


### Subnational

#### District Data

In this file are summarized the socioeconomic and spatial data on economic production for each district. A GeoJSON file is expected.
- `district_code`: a string with the administrative code of the district
- `geometry`: a 'Point', which represent where the firms and household modelled for this district will be located. Typically the location of the largest city in the district, or, as a second best option, the district's centroid.
- `population`: the number of people in the district
- economic supply data: at least one column per sector which captures the size of the sector in this district. For instance, there could be the number of employees in the manufacturing of basic metals, the total sales of construction, and the value of production of agricultural products. The column name should corresponds to the values of the 'supply_data' column in the 'sector firm cutoff' file

A GeoJSON file `economic_data.geojson` with Points and the following attribute table:

Example:

district | nb_workers_MAN | nb_workers_ELE | crop_production | ... | geometry
--- | --- | --- | --- | --- | ---
0101 | 124 | 12 | 465120 | ... | Point(-45.15631, 0.48654)
... | ... | ... | ... | ... | ...



### Trade

#### Country Transit Matrix

A CSV file representing a matrix. Country codes are row and column headers. It represents the yearly trade flow from the row country to the column country that goes through the country-under-study's transport network. The unit should be the same as defined in the parameter `monetary_unit_in_model`.

Example:

... | BDI | COD | ...
--- | --- | --- | --- 
**BDI** | 4563 | 4516 | ...
... | ... | ... | ...

#### Import Table

A CSV file representing a matrix. Country codes are row headers. Sector codes are column headers. It represents the yearly imports from the different countries, in row, to the domestic sectors, in column. The unit should be the same as defined in the parameter `monetary_unit_in_model`.

Example:

... | AGR | FOR | ...
--- | --- | --- | --- 
**BDI** | 132 | 0 | ...
... | ... | ... | ...


#### Export Table

A CSV file representing a matrix. Country codes are row headers. Sector codes are column headers. It represents the yearly exports from the domestic sectors, in columns, to the foreigh countries, in row. The unit should be the same as defined in the parameter `monetary_unit_in_model`.

Example:

... | AGR | FOR | ...
--- | --- | --- | --- 
**BDI** | 2 | 456 | ...
... | ... | ... | ...

