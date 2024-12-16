import logging

import pandas as pd


def compare_production_purchase_plans(firms, countries, households):
    # Create dictionary to map firm id and country id into sector
    dic_agent_id_to_sector = {pid: firm.sector for pid, firm in firms.items()}
    for pid, country in countries.items():
        dic_agent_id_to_sector[pid] = "IMP"

    # Evaluate purchase plans of firms
    df = pd.DataFrame({pid: firm.purchase_plan for pid, firm in firms.items()})
    df["tot_purchase_planned_by_firms"] = df.sum(axis=1)
    df['input_sector'] = df.index.map(dic_agent_id_to_sector)
    df_firms = df.groupby('input_sector')["tot_purchase_planned_by_firms"].sum()

    # of countries
    if len(countries) == 1:
        country_id = list(countries.keys())[0]
        df = pd.DataFrame(pd.Series(countries[country_id].purchase_plan, name=country_id))
    df = pd.DataFrame({pid: country.purchase_plan for pid, country in countries.items()})
    df["tot_purchase_planned_by_countries"] = df.sum(axis=1)
    df['input_sector'] = df.index.map(dic_agent_id_to_sector)
    df_countries = df.groupby('input_sector')["tot_purchase_planned_by_countries"].sum()
    # of households
    df = pd.DataFrame({pid: household.purchase_plan for pid, household in households.items()})
    df["tot_purchase_planned_by_households"] = df.sum(axis=1)
    # df = pd.DataFrame({"tot_purchase_planned_by_households": households.purchase_plan})
    df['input_sector'] = df.index.map(dic_agent_id_to_sector)
    df_households = df.groupby('input_sector')["tot_purchase_planned_by_households"].sum()

    # concat
    df_purchase_plan = pd.concat([df_firms, df_countries, df_households], axis=1, sort=True)

    # Evalute productions/sales
    # of firms
    df = pd.DataFrame({
        "tot_production_per_firm": {pid: firm.production for pid, firm in firms.items()}
    }
    )
    df['sector'] = df.index.map(dic_agent_id_to_sector)
    df_firms = df.groupby('sector')["tot_production_per_firm"].sum()

    # of countries
    df = pd.DataFrame(
        {
            "tot_production_per_country":
                {pid: country.qty_sold for pid, country in countries.items()}
        }
    )
    df['sector'] = df.index.map(dic_agent_id_to_sector)
    df_countries = df.groupby('sector')["tot_production_per_country"].sum()

    # concat
    df_sales = pd.concat([df_firms, df_countries], axis=1, sort=True)

    # Compare
    res = pd.concat([df_purchase_plan, df_sales], axis=1, sort=True)
    res['dif'] = (res["tot_purchase_planned_by_firms"]
                  + res["tot_purchase_planned_by_countries"]
                  + res["tot_purchase_planned_by_households"]
                  - res['tot_production_per_firm'] - res['tot_production_per_country'])
    boolindex_unbalanced = res['dif'] > 1e-6
    if boolindex_unbalanced.sum() > 0:
        logging.warning("Sales does not equate purchases for sectors: " +
                        str(res.index[boolindex_unbalanced].tolist()))


# def compareDeliveredVsReceived(firm_list=None, households=None, G=None):
#     # not finished
#     qty_delivered_by_firm_per_sector = {}
#     for firm in firm_list:
#         if firm.sector not in qty_delivered_by_firm_per_sector.keys():
#             qty_delivered_by_firm_per_sector[firm.sector] = 0
#
#         for edge in G.out_edges(firm):
#             qty_delivered_by_firm_per_sector[firm.sector] += \
#                 G[firm][edge[1]]['object'].delivery
#
#         qty_bought_by_household_per_sector = {}
#         for edge in G.in_edges(households):
#             if edge[0].sector not in qty_bought_by_household_per_sector.keys():
#                 qty_bought_by_household_per_sector[edge[0].sector] = 0
#             qty_bought_by_household_per_sector[firm.sector] += \
#                 G[edge[0]][households]['object'].delivery
#
#     qty_ordered_by_firm_per_sector = {}
#     for firm in firm_list:
#         if firm.sector not in qty_ordered_by_firm_per_sector.keys():
#             qty_delivered_by_firm_per_sector[firm.sector] = 0
