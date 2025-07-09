import pandas as pd

from disruptsc.network.route import Route
from disruptsc.parameters import EPSILON


class CommercialLink(object):

    def __init__(self, pid=None, supplier_id=None, buyer_id=None, product=None,
                 origin_node=None, destination_node = None,
                 product_type=None, category=None, order=0, delivery=0, payment=0, essential=True,
                 route: "Route" = None):
        # Parameter
        self.pid = pid
        self.product = product  # sector of producing firm
        self.product_type = product_type  # service, manufacturing, etc. (=sector_type)
        self.category = category  # import, export, domestic_B2B, transit
        self.route = route or []  # node_id path of the transport network, as
        # [(node1, ), (node1, node2), (node2, ), (node2, node3), (node3, )]
        self.route_length = 1
        self.route_cost_per_ton = 0
        self.supplier_id = supplier_id
        self.buyer_id = buyer_id
        self.origin_node = origin_node
        self.destination_node = destination_node
        self.eq_price = 1
        self.shipment_method = "solid_bulk"
        self.essential = essential
        self.use_transport_network = False

        # Variable
        self.alternative_found = False
        self.status = "ok"
        self.current_route = 'main'
        self.order = order  # flows upstream
        self.delivery = delivery  # flows downstream. What is supposed to be delivered (if no transport pb)
        self.delivery_in_tons = delivery  # flows downstream. What is supposed to be delivered (if no transport pb)
        self.realized_delivery = delivery
        self.payment = payment  # flows upstream
        self.alternative_route = None
        self.alternative_route_length = 1
        self.alternative_route_cost_per_ton = 0
        self.price = 1
        self.fulfilment_rate = 1  # ratio deliver / order

    def get_current_route(self):
        if self.current_route == "main":
            return self.route
        elif self.current_route == "alternative":
            return self.alternative_route
        else:
            ValueError("current_route should be 'main' or 'alternative'")

    def update_status(self):
        delivery = "ok"
        price = "ok"
        if abs(self.price - self.eq_price) > EPSILON:
            price = "more expensive"
        if (self.fulfilment_rate > EPSILON) and (self.fulfilment_rate < 1 - EPSILON):
            delivery = "partial"
        elif self.fulfilment_rate < EPSILON:
            delivery = "no delivery"

        if (delivery == "ok") and (price == "ok"):
            self.status = "ok"
        else:
            self.status = f"delivery: {delivery}, price: {price}"

    def determine_transportation_mode(self, sector_types_to_shipment_method: dict):
        if self.product_type in sector_types_to_shipment_method.keys():
            self.shipment_method = sector_types_to_shipment_method[self.product_type]
        else:
            self.shipment_method = sector_types_to_shipment_method['default']

    def print_info(self):
        # print("\nCommercial Link from "+str(self.supplier_id)+" to "+str(self.buyer_id)+":")
        # print("route:", self.route)
        # print("alternative route:", self.alternative_route)
        # print("product:", self.product)
        # print("order:", self.order)
        # print("delivery:", self.delivery)
        # print("payment:", self.payment)
        attribute_to_print = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]
        for attribute in attribute_to_print:
            print(attribute + ": " + str(getattr(self, attribute)))

    def reset_variables(self):
        # Variable
        self.current_route = 'main'
        self.order = 0  # flows upstream
        self.delivery = 0  # flows downstream
        self.payment = 0  # flows upstream
        self.fulfilment_rate = 1
        self.alternative_route = []
        self.alternative_route_cost_per_ton = 0
        self.price = 1
        self.alternative_found = False
        self.status = "ok"

    def calculate_fulfilment_rate(self):
        if self.order < EPSILON:
            self.fulfilment_rate = 1
        elif self.delivery > self.order + EPSILON:
            self.fulfilment_rate = 1
        else:
            self.fulfilment_rate = self.delivery / self.order

    def calculate_relative_increase_in_transport_cost(self):
        if self.delivery_in_tons == 0:
            raise ValueError('Delivery in tons is null')
        normal_transport_bill = self.delivery_in_tons * self.route_cost_per_ton
        new_transport_bill = self.delivery_in_tons * self.alternative_route_cost_per_ton
        return max(new_transport_bill - normal_transport_bill, 0) / normal_transport_bill

    def has_modal_switch(self):
        """Check if alternative route uses different transportation modes than main route."""
        if not self.alternative_found:
            return False
        main_modes = set(self.route.transport_modes)
        alt_modes = set(self.alternative_route.transport_modes)
        return main_modes != alt_modes

    def has_port_switch(self, transport_network):
        """Check if alternative route uses different maritime multimodal edges than main route."""
        if not self.alternative_found:
            return False
        
        main_maritime_edges = self.route.get_maritime_multimodal_edges(transport_network)
        alt_maritime_edges = self.alternative_route.get_maritime_multimodal_edges(transport_network)
        
        # Return True if both routes use maritime multimodal edges but different ones
        return (len(main_maritime_edges) > 0 and 
                len(alt_maritime_edges) > 0 and 
                main_maritime_edges != alt_maritime_edges)

    def calculate_switching_cost(self, switching_costs: dict, transport_network):
        """Calculate additional percentage cost penalty for switching transportation modes."""
        if self.has_modal_switch():
            return switching_costs['modal_switch']
        elif self.has_port_switch(transport_network):
            return switching_costs['port_switch']
        return 0

    def store_route_information(self, route: Route, main_or_alternative: str, cost_per_ton: float):

        self.use_transport_network = True

        if main_or_alternative == "main":
            self.route = route
            self.route_length = route.length
            self.route_cost_per_ton = cost_per_ton

        elif main_or_alternative == "alternative":
            self.alternative_found = True
            self.alternative_route = route
            self.alternative_route_length = route.length
            self.alternative_route_cost_per_ton = cost_per_ton

        else:
            raise ValueError("'main_or_alternative' is not in ['main', 'alternative']")
