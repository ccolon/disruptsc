class Shipment(dict):
    def __init__(self, commercial_link_id, arg2=None, **kwargs):
        # Ensure arg1 is always provided
        if commercial_link_id is None:
            raise ValueError("commercial_link_id is required")

        # Initialize the dictionary with the provided arguments
        super().__init__(arg1=arg1, arg2=arg2, **kwargs)

    def __repr__(self):
        return f"Shipment({super().__repr__()})"