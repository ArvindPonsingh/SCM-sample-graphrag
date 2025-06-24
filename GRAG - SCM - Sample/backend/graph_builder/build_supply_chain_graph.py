import pandas as pd
import networkx as nx
import os

def load_data_and_build_graph(
    products_file,
    supplier_info_file,
    shipping_records_file
):
    graph = nx.DiGraph()

    try:
        products_df = pd.read_csv(products_file)
        supplier_df = pd.read_csv(supplier_info_file)
        shipping_df = pd.read_csv(shipping_records_file)

        location_ids = set(shipping_df['Source']).union(
            set(shipping_df['Destination'])
        )
        for loc_id in location_ids:
            if loc_id not in graph:
                graph.add_node(loc_id, type='Location', Location_ID=loc_id, Name=f"Location {loc_id}")

        for _, row in products_df.iterrows():
            product_id = row['ProductID']
            graph.add_node(product_id, type='Product', **row.to_dict())

        for _, row in supplier_df.iterrows():
            supplier_id = row['SupplierID']
            graph.add_node(supplier_id, type='Supplier', **row.to_dict())

        for _, row in shipping_df.iterrows():
            shipment_id = row['ShipmentID']
            graph.add_node(shipment_id, type='Shipment', **row.to_dict())
            
        for _, row in shipping_df.iterrows():
            graph.add_edge(
                row['Source'], row['ShipmentID'],
                relation='origin_for'
            )
            graph.add_edge(
                row['ShipmentID'], row['Destination'],
                relation='destination_is'
            )
            
            if 'ProductID' in row and pd.notna(row['ProductID']):
                graph.add_edge(
                    row['ShipmentID'], row['ProductID'],
                    relation='transports_product'
                )
            
            if 'SupplierID' in row and pd.notna(row['SupplierID']):
                graph.add_edge(
                    row['SupplierID'], row['ShipmentID'],
                    relation='facilitates_shipment'
                )

        return graph

    except FileNotFoundError as e:
        return None
    except Exception as e:
        return None

# if __name__ == '__main__':
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
#     data_dir = os.path.join(project_root, 'data') 

#     products_file = os.path.join(data_dir, 'products.csv')
#     supplier_info_file = os.path.join(data_dir, 'supplier_info.csv')
#     shipping_records_file = os.path.join(data_dir, 'shipping_records.csv')

#     supply_chain_graph = load_data_and_build_graph(
#         products_file,
#         supplier_info_file,
#         shipping_records_file
#     )

#     if supply_chain_graph:
#         pass
#     else:
#         pass