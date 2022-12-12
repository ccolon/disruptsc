class Agent(object):
    def __init__(self, agent_type, pid, odpoint=0,
                 long=None, lat=None):
        self.agent_type = agent_type
        self.pid = pid
        self.odpoint = odpoint
        self.long = long
        self.lat = lat
        self.usd_per_ton = None

    def choose_initial_routes(self, sc_network, transport_network, transport_modes,
                              account_capacity, monetary_unit_flow):
        for edge in sc_network.out_edges(self):
            if edge[1].pid == -1:  # we do not create route for households
                continue
            elif edge[1].odpoint == -1:  # we do not create route for service firms if explicit_service_firms = False
                continue
            else:
                # Get the id of the orign and destination node
                origin_node = self.odpoint
                destination_node = edge[1].odpoint
                cond_from, cond_to = self.get_transport_cond(edge, transport_modes)
                transport_mode = transport_modes.loc[cond_from & cond_to, "transport_mode"].iloc[0]
                sc_network[self][edge[1]]['object'].transport_mode = transport_mode
                # Choose the route and the corresponding mode
                route, selected_mode = self.choose_route(
                    transport_network=transport_network,
                    origin_node=origin_node,
                    destination_node=destination_node,
                    accepted_logistics_modes=transport_mode
                )
                # print(str(self.pid)+" located "+str(self.odpoint)+": I choose this transport mode "+
                #     str(transport_network.give_route_mode(route))+ " to connect to "+
                #     str(edge[1].pid)+" located "+str(edge[1].odpoint))
                # Store it into commercial link object
                sc_network[self][edge[1]]['object'].storeRouteInformation(
                    route=route,
                    transport_mode=selected_mode,
                    main_or_alternative="main",
                    transport_network=transport_network
                )

                if account_capacity:
                    self.update_transport_load(edge, monetary_unit_flow, route, sc_network, transport_network)

    def get_transport_cond(self, edge, transport_modes):
        # Define the type of transport mode to use by looking in the transport_mode table
        if self.agent_type == 'firm':
            cond_from = (transport_modes['from'] == "domestic")
        elif self.agent_type == 'country':
            cond_from = (transport_modes['from'] == self.pid)
        else:
            raise ValueError("'self' must be a Firm or a Country")
        if edge[1].agent_type in ['firm', 'household']:  # see what is the other end
            cond_to = (transport_modes['to'] == "domestic")
        elif edge[1].agent_type == 'country':
            cond_to = (transport_modes['to'] == edge[1].pid)
        else:
            raise ValueError("'edge[1]' must be a Firm or a Country")
            # we have not implemented a "sector" condition
        return cond_from, cond_to

    def update_transport_load(self, edge, monetary_unit_flow, route, sc_network, transport_network):
        # Update the "current load" on the transport network
        # if current_load exceed burden, then add burden to the weight
        new_load_in_usd = sc_network[self][edge[1]]['object'].order
        new_load_in_tons = Agent.transformUSD_to_tons(new_load_in_usd, monetary_unit_flow, self.usd_per_ton)
        transport_network.update_load_on_route(route, new_load_in_tons)

    def choose_route(self, transport_network, origin_node, destination_node, accepted_logistics_modes):
        raise NotImplementedError

    @staticmethod
    def check_route_availability(commercial_link, transport_network, which_route='main'):
        """
        Look at the main or alternative route
        at check all edges and nodes in the route
        if one is marked as disrupted, then the whole route is marked as disrupted
        """

        if which_route == 'main':
            route_to_check = commercial_link.route
        elif which_route == 'alternative':
            route_to_check = commercial_link.alternative_route
        else:
            raise KeyError('Wrong value for parameter which_route, admissible values are main and alternative')

        res = 'available'
        for route_segment in route_to_check:
            if len(route_segment) == 2:
                if transport_network[route_segment[0]][route_segment[1]]['disruption_duration'] > 0:
                    res = 'disrupted'
                    break
            if len(route_segment) == 1:
                if transport_network._node[route_segment[0]]['disruption_duration'] > 0:
                    res = 'disrupted'
                    break
        return res

    @staticmethod
    def transformUSD_to_tons(monetary_flow, monetary_unit, usd_per_ton):
        if usd_per_ton == 0:
            return 0
        else:
            # Load monetary units
            monetary_unit_factor = {
                "mUSD": 1e6,
                "kUSD": 1e3,
                "USD": 1
            }
            factor = monetary_unit_factor[monetary_unit]
            return monetary_flow / (usd_per_ton / factor)
