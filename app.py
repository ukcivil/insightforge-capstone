# All Import Commands needed for Functionality
import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

# Uses Streamlit secrets to secure my API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Page configuration and setup
st.set_page_config(page_title="InsightForge Business Intelligence (BI) Assistant", layout="wide")
st.title("InsightForge – AI-Powered Business Intelligence Assistant")

# Allows upload of CSV file
uploaded_file = st.sidebar.file_uploader("Upload Sales Data CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data Loaded")

    # Prepares base documents from raw data
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M').astype(str)

    documents = []
    for _, row in df.iterrows():
        doc = {
            "date": row['Date'].strftime('%Y-%m-%d'),
            "product": row['Product'],
            "region": row['Region'],
            "sales": row['Sales'],
            "customer_age": row['Customer_Age'],
            "gender": row['Customer_Gender'],
            "satisfaction": round(row['Customer_Satisfaction'], 2)
        }
        text = f"""
PRODUCT REPORT

Date: {doc['date']}
Product: {doc['product']}
Region: {doc['region']}
Sales: {doc['sales']}
Customer Age: {doc['customer_age']}
Gender: {doc['gender']}
Satisfaction: {doc['satisfaction']}
"""
        documents.append(Document(page_content=text))

    # Add summaries to allow for more complete insight coverage

    # Product by region
    prod_region = df.groupby(['Product', 'Region'])['Sales'].sum().reset_index()
    for _, row in prod_region.iterrows():
        documents.append(Document(page_content=f"Product: {row['Product']}, Region: {row['Region']}, Total Sales: {row['Sales']}"))

    # Product by month
    prod_month = df.groupby(['Product', 'Month'])['Sales'].sum().reset_index()
    for _, row in prod_month.iterrows():
        documents.append(Document(page_content=f"Product: {row['Product']} had {row['Sales']} sales in {row['Month']}"))

    # Product satisfaction
    prod_sat = df.groupby('Product')['Customer_Satisfaction'].mean().reset_index()
    for _, row in prod_sat.iterrows():
        documents.append(Document(page_content=f"Product: {row['Product']}, Avg Satisfaction: {round(row['Customer_Satisfaction'], 2)}"))

    # Region age and sales summary
    region_agg = df.groupby('Region').agg({'Customer_Age': 'mean', 'Sales': 'sum'}).reset_index()
    for _, row in region_agg.iterrows():
        documents.append(Document(page_content=f"Region: {row['Region']}, Avg Age: {round(row['Customer_Age'], 1)}, Total Sales: {row['Sales']}"))

    # Monthly total sales
    monthly = df.groupby('Month')['Sales'].sum().reset_index()
    for _, row in monthly.iterrows():
        documents.append(Document(page_content=f"Monthly Summary - {row['Month']}: Total Sales = {row['Sales']}"))

    # RAG system setup
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # FAQ list provided to user for further guidance.
    
    with st.expander("Suggested Typical Questions and Data Analysis Request"):
        st.markdown("""
    ### Sales Trends
    - What is the overall sales trend over time?
    - Compare monthly sales across 2023.
    - Was there a seasonal spike in 2024?

    ### Product Performance
    - Compare total sales for Widget A and Widget B.
    - Which product had the highest sales in 2023?
    - How did Widget C perform across regions?

    ### Regional Insights
    - Which region had the highest total sales?
    - Compare Widget A and Widget B sales by region.
    - What is the average customer age in the South?

    ### Customer Satisfaction
    - What is the average satisfaction score for Widget B?
    - Compare satisfaction scores between products.
    - How does customer satisfaction vary by region?

    ###  Customer Demographics
    - Which region has the youngest customers?
    - How do male and female customers differ in purchases?
    """)
    
    # Input query displayed to the user
    user_query = st.text_input("Ask a business question or request data analysis:")

    if user_query:
        result = qa_chain.invoke({"query": user_query})
        #  Show AI response
        st.subheader(" AI Insight")
        st.write(result["result"])

        # Warn if no documents were retrieved
        if not result.get("source_documents"):
            st.warning("I couldn’t find enough data to confidently answer that question. Try rewording it or check if the dataset includes that info.")

        # Flag questions that might require unsupported logic
        unsupported_patterns = [
            "average sales per transaction",
            "total transactions in",
            "most recent sale",
            "top customer",
            "sales in the last",  # time-based filters
            "profit",             # not in dataset
        ]

        if any(p in user_query.lower() for p in unsupported_patterns):
            st.warning("This type of question may not be fully supported by the dataset or summaries. Results could be limited.")

        # Optional: Let user view the source data used for the answer
        if st.checkbox("Show retrieved data"):
            st.markdown("These are the data pieces the AI used to generate your answer:")
            for doc in result["source_documents"]:
                st.markdown(f"• {doc.page_content.strip()}")
        # Creates the interaction log
        with open("chat_log.txt", "a") as log_file:
            log_file.write(f"Time: {datetime.now().isoformat()}\n")
            log_file.write(f"User Query: {user_query}\n")
            log_file.write(f"AI Response: {result['result']}\n")
            log_file.write("-" * 50 + "\n")

        # Reduces irrelevant visuals
        q = user_query.lower()

        if "sales trend" in q or "monthly sales" in q or "over time" in q:
            monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=monthly_sales, x='Month', y='Sales', marker='o', ax=ax)
            ax.set_title('Monthly Sales Trend')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        elif "sales by product" in q or "compare products" in q:
            fig, ax = plt.subplots()
            sns.barplot(data=df, x='Product', y='Sales', estimator=sum, ax=ax)
            ax.set_title('Total Sales by Product')
            st.pyplot(fig)

        elif "satisfaction" in q and "distribution" in q:
            fig, ax = plt.subplots()
            sns.histplot(df['Customer_Satisfaction'], bins=20, kde=True, ax=ax)
            ax.set_title('Customer Satisfaction Distribution')
            st.pyplot(fig)

# Prompt to upload file as a CSV file
else:
    st.warning("Please upload a CSV file to begin.")

# Creates the button allowing download of the interaction log
if os.path.exists("chat_log.txt"):
    with open("chat_log.txt", "r") as log_file:
        st.download_button("Download Interaction Log", log_file, file_name="chat_log.txt")
