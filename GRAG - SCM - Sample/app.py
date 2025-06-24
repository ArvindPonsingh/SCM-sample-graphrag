import streamlit as st
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(project_root, "backend")
sys.path.insert(0, backend_path) 

from backend.graph_builder.build_supply_chain_graph import load_data_and_build_graph
from backend.retriever.find_relevant_info import find_relevant_info, encode_node_attributes 
from backend.Groq_integration.ask_Groq import get_groq_answer 

st.set_page_config(
    page_title="Supply Chain Chatbot",
    page_icon="📦",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("📦 Supply Chain Management Chatbot")
st.markdown("Ask questions about your supply chain data!")

@st.cache_resource
def load_graph_and_prepare_embeddings():
    st.info("Loading supply chain graph and preparing for queries...")

    products_file = r'C:\Users\ADMIN\Desktop\4i doc\GraphRAG SupplyChain BOT\GRAG - SCM - Sample\data\products.csv'
    supplier_info_file = r'C:\Users\ADMIN\Desktop\4i doc\GraphRAG SupplyChain BOT\GRAG - SCM - Sample\data\suppliers.csv'
    shipping_records_file = r'C:\Users\ADMIN\Desktop\4i doc\GraphRAG SupplyChain BOT\GRAG - SCM - Sample\data\shipments.csv'

    graph = load_data_and_build_graph(
        products_file,
        supplier_info_file,
        shipping_records_file
    )
    
    if not graph:
        st.error("Error: Failed to load the supply chain graph. Please check your data files/URLs and ensure they are correct.")
        return None, None
        
    st.success("Supply Chain Graph loaded successfully.")
    
    st.info("Encoding graph nodes for faster retrieval...")
    node_embeddings = encode_node_attributes(graph) 
    st.success("Node embeddings computed!")
    
    return graph, node_embeddings

supply_chain_graph, node_embeddings = load_graph_and_prepare_embeddings()

if supply_chain_graph is None or node_embeddings is None:
    st.warning("Chatbot functionality is limited because the graph could not be loaded or embeddings could not be prepared. Please check data files/URLs and your console for errors.")
else:
    query = st.text_input("Enter your question:", key="user_query_input")

    if st.button("Get Answer"):
        if query:
            try:
                relevant_context = find_relevant_info(supply_chain_graph, query, precomputed_node_embeddings=node_embeddings)
                
                context_for_llm = relevant_context

                st.write("\n**Chatbot's Answer:**")
                with st.spinner("Thinking..."): 
                    answer = get_groq_answer(query, context_for_llm) 
                st.markdown(answer)

            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
                st.exception(e)

        else:
            st.warning("Please enter a question to get an answer.")