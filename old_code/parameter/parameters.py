from old_code.parameter.parameters_default import *
import logging

# TODO make it a dictionary
# TODO create a function to merge the default param dic and the param dic

input_folder = "Ecuador"
# inventory_duration_target = "inputed"

logging_level = logging.INFO

monetary_units_in_model = "mUSD"
monetary_units_inputed = "kUSD"
time_resolution = 'week'

firm_data_type = "disaggregating IO"  # disaggregating IO, supplier-buyer network

transport_modes = ['roads', 'maritime', 'waterways', 'airways']

logistics_modes = {
    'domestic': {
        "accepted_modes": ['roads', 'airways', 'maritime', 'waterways', 'multimodal'],
        "accepted_multimodal_links": ['roads-maritime-dom', 'roads-airways', 'roads-waterways']
    },
    'international': {
        "accepted_modes": ['roads', 'airways', 'maritime', 'waterways', 'multimodal'],
        "accepted_multimodal_links": ['roads-maritime', 'roads-maritime-dom', 'roads-airways', 'roads-waterways']
    }
}

route_optimization_weight = "cost_per_ton"  # cost_per_ton

pop_density_cutoff = 0
pop_cutoff = 1000
local_demand_cutoff = 50

# disruption_analysis = {
#     "type": "criticality",
#     "disrupt_nodes_or_edges": "edges",
#     "nodeedge_tested": "all",
#     "identified_by": "id",
#     "start_time": 1,
#     "duration": 1
# }

affected_sectors = ['ALD', 'AYG', 'AZU', 'BAL', 'BNA', 'CAN', 'CAR', 'CAU', 'CEM',
                    'CHO', 'COM', 'CON', 'CUE', 'DEM', 'EDU', 'ELE', 'FID', 'HIL',
                    'HOT', 'LAC', 'MAD', 'MAN', 'MAQ', 'MET', 'MIP', 'MOL', 'MUE',
                    'PAN', 'PAP', 'POS', 'PPR', 'QU1', 'QU2', 'REF', 'REP', 'RES',
                    'TEL', 'TRA', 'VES', 'VID']

utilization_rate = 1

disruption_analysis = {
    "type": "compound",
    "events": [
        # {
        #     "item_type": "transport_edges",
        #     "attribute": "disruption",
        #     # "attribute": "id",
        #     "values": ["2016_earthquake"],
        #     # "values": [1200],
        #     "start_time": 1,
        #     "duration": 1
        # }#,
        {
            "item_type": "firms",
            "admin_units": ["1309"],
            "sectors": affected_sectors,
            "start_time": 1,
            "duration": 2,
            "production_capacity_reduction": 0.004153
        },
        {
            "item_type": "firms",
            "admin_units": ["1301"],
            "sectors": affected_sectors,
            "start_time": 1,
            "duration": 2,
            "production_capacity_reduction": 0.006068
        },
        {
            "item_type": "firms",
            "admin_units": ["1308"],
            "sectors": affected_sectors,
            "start_time": 1,
            "duration": 2,
            "production_capacity_reduction": 0.007499
        },
        {
            "item_type": "firms",
            "admin_units": ["1321"],
            "sectors": affected_sectors,
            "start_time": 1,
            "duration": 2,
            "production_capacity_reduction": 0.010603
        },
        {
            "item_type": "firms",
            "admin_units": ["1303"],
            "sectors": affected_sectors,
            "start_time": 1,
            "duration": 2,
            "production_capacity_reduction": 0.013404
        },
        {
            "item_type": "firms",
            "admin_units": ["1322"],
            "sectors": affected_sectors,
            "start_time": 1,
            "duration": 2,
            "production_capacity_reduction": 0.022554
        },
        {
            "item_type": "firms",
            "admin_units": ["803"],
            "sectors": affected_sectors,
            "start_time": 1,
            "duration": 2,
            "production_capacity_reduction": 0.116392
        },
        {
            "item_type": "firms",
            "admin_units": ["1320"],
            "sectors": affected_sectors,
            "start_time": 1,
            "duration": 2,
            "production_capacity_reduction": 0.177236
        },
        {
            "item_type": "firms",
            "admin_units": ["1317"],
            "sectors": affected_sectors,
            "start_time": 1,
            "duration": 2,
            "production_capacity_reduction": 0.420373
        }
    ]
}

disruption_analysis = {
    "type": "criticality",
    "disrupt_nodes_or_edges": "edges",
    "nodeedge_tested": "all",
    "identified_by": "id",
    "start_time": 1,
    "duration": 1
}
# disruption_analysis = None
# inventory_duration_target = 2
# congestion = True

# cutoffs
# sectors_to_exclude = ['ADM']
# district_sector_cutoff = 0.003
# cutoff_sector_output = {
#     'type': 'percentage',
#     'value': 0.02
# }
io_cutoff = 0.01

route_optimization_weight = "cost_per_ton"  # cost_per_ton time_cost agg_cost

export = {key: True for key in export.keys()}
# export['transport'] = True

cost_repercussion_mode = "type1"

# duration_dic[1] = 1


export = {
    # Save a log file in the output folder, called "exp.log"
    "log": True,

    # Transport nodes and edges as geojson
    "transport": True,

    # Save the main result in a "criticality.csv" file in the output folder
    # Each line is a simulation, it saves what is disrupted and for how long, and aggregate observables
    "criticality": True,

    # Save the amount of good flowing on each transport segment
    # It saves a flows.json file in the output folder
    # The structure is a dic {"timestep: {"transport_link_id: {"sector_id: flow_quantity}}
    # Can be True or False
    "flows": True,

    # Export information on aggregate supply chain flow at initial conditions
    # Used only if "disruption_analysis: None"
    # See analyzeSupplyChainFlows function for details
    "sc_flow_analysis": False,

    # Whether or not to export data for each agent for each time steps
    # See exportAgentData function for details.
    "agent_data": True,

    # Save firm-level impact results
    # It creates an "extra_spending.csv" file and an "extra_consumption.csv" file in the output folder
    # Each line is a simulation, it saves what was disrupted and the corresponding impact for each firm
    "impact_per_firm": True,

    # Save aggregated time series
    # It creates an "aggregate_ts.csv" file in the output folder
    # Each columns is a time series
    # Exports:
    # - aggregate production
    # - total profit, 
    # - household consumption, 
    # - household expenditure, 
    # - total transport costs, 
    # - average inventories.
    "time_series": False,

    # Save the firm table
    # It creates a "firm_table.xlsx" file in the output folder
    # It gives the properties of each firm, along with production, sales to households, to other firms, exports
    "firm_table": True,

    # Save the OD point table
    # It creates a "odpoint_table.xlsx" file in the output folder
    # It gives the properties of each OD point, along with production, sales to households, to other firms, exports
    "odpoint_table": True,

    # Save the country table
    # It creates a "country_table.xlsx" file in the output folder
    # It gives the trade profile of each country
    "country_table": True,

    # Save the edgelist table
    # It creates a "edgelist_table.xlsx" file in the output folder
    # It gives, for each supplier-buyer link, the distance and amounts of good that flows
    "edgelist_table": False,

    # Save inventories per sector
    # It creates an "inventories.xlsx" file in the output folder
    "inventories": False,

    # Save the combination of district and sector that are over the cutoffs value
    # It creates an "filtered_district_sector.xlsx" file in the output folder
    "district_sector_table": False,

    # Whether or not to export a csv summarizing some topological caracteristics of the supply chain network
    "sc_network_summary": False
}


def create_dict(*args):
    return dict(((k, eval(k)) for k in args))


parameters = create_dict()
