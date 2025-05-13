# 📦 Imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import os

# 🔐 Load API Key securely from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ⚙️ Streamlit page config
st.set_page_config(page_title="InsightForge BI Assistant", layout="wide")
st.title("\U0001F4CA InsightForge – AI-Powered Business Intelligence Assistant")

# 📁 Upload CSV from sidebar
uploaded_file = st.sidebar.file_uploader("Upload Sales Data CSV", type="csv")

if uploaded_file:
    # ✅ Load and prepare data
    df = pd.read_csv(uploaded_file)
    st.success("\u2705 Data Loaded")

    # 🧠 Step 1: Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M').astype(str)

    # 📝 Step 2: Create document list from raw records
    documents = []
    for _, row in df.iterrows():
        doc = f"""
PRODUCT REPORT

Date: {row['Date'].strftime('%Y-%m-%d')}
Product: {row['Product']}
Region: {row['Region']}
Sales: {row['Sales']}
Customer Age: {row['Customer_Age']}
Gender: {row['Customer_Gender']}
Satisfaction: {round(row['Customer_Satisfaction'], 2)}
"""
        documents.append(Document(page_content=doc))

    # 📊 Step 3: Add Monthly Sales Summaries
    monthly_summary = df.groupby('Month')['Sales'].sum()
    for month, total in monthly_summary.items():
        documents.append(Document(page_content=f"Monthly Summary - {month}: Total Sales = {total}"))

    # 🌍 Step 4: Add Product-Region Summaries
    product_region = df.groupby(['Product', 'Region'])['Sales'].sum().reset_index()
    for _, row in product_region.iterrows():
        doc = f"Product: {row['Product']}, Region: {row['Region']}, Total Sales = {row['Sales']}"
        documents.append(Document(page_content=doc))

    # 😊 Step 5: Add Product Satisfaction Summaries
    prod_sat = df.groupby('Product')['Customer_Satisfaction'].mean().reset_index()
    for _, row in prod_sat.iterrows():
        doc = f"Product: {row['Product']}, Avg Satisfaction = {round(row['Customer_Satisfaction'], 2)}"
        documents.append(Document(page_content=doc))

    # 👥 Step 6: Add Region Age + Sales Summaries
    region_demo = df.groupby('Region').agg({'Customer_Age': 'mean', 'Sales': 'sum'}).reset_index()
    for _, row in region_demo.iterrows():
        doc = f"Region: {row['Region']}, Avg Age = {round(row['Customer_Age'], 1)}, Total Sales = {row['Sales']}"
        documents.append(Document(page_content=doc))

    # 🧠 Step 7: Embed documents and create retriever
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # 💬 Step 8: User query input
    user_query = st.text_input("\U0001F4AC Ask a business question:")

    if user_query:
        result = qa_chain.invoke({"query": user_query})
        st.subheader("\U0001F9E0 AI Insight")
        st.write(result["result"])

        # 🪵 Log interaction
        with open("chat_log.txt", "a") as log_file:
            log_file.write(f"Time: {datetime.now().isoformat()}\n")
            log_file.write(f"User Query: {user_query}\n")
            log_file.write(f"AI Response: {result['result']}\n")
            log_file.write("-" * 50 + "\n")

        # 📊 Step 9: Trigger matching visualizations
        q = user_query.lower()

        if "sales trend" in q or "sales over time" in q or "monthly sales" in q or "sales by month" in q:
            trend = df.groupby('Month')['Sales'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=trend, x='Month', y='Sales', marker='o', ax=ax)
            ax.set_title('Monthly Sales Trend')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        elif "sales by product" in q or "product performance" in q:
            fig, ax = plt.subplots()
            sns.barplot(data=df, x='Product', y='Sales', estimator=sum, ax=ax)
            ax.set_title('Total Sales by Product')
            st.pyplot(fig)

        elif "sales by region" in q or "regional sales" in q:
            fig, ax = plt.subplots()
            sns.barplot(data=df, x='Region', y='Sales', estimator=sum, ax=ax)
            ax.set_title('Total Sales by Region')
            st.pyplot(fig)

        elif "customer age" in q or "age distribution" in q:
            fig, ax = plt.subplots()
            sns.histplot(df['Customer_Age'], bins=10, kde=True, ax=ax)
            ax.set_title('Customer Age Distribution')
            st.pyplot(fig)

        elif "satisfaction" in q and "distribution" in q:
            fig, ax = plt.subplots()
            sns.histplot(df['Customer_Satisfaction'], bins=20, kde=True, ax=ax)
            ax.set_title('Customer Satisfaction Distribution')
            st.pyplot(fig)

        elif "compare" in q and "widget" in q:
            products = ["Widget A", "Widget B", "Widget C", "Widget D"]
            subset = df[df['Product'].isin(products)]
            fig, ax = plt.subplots()
            sns.boxplot(data=subset, x='Product', y='Sales', ax=ax)
            ax.set_title("Sales Comparison Across Products")
            st.pyplot(fig)

else:
    # ⚠️ Show warning if no file uploaded
    st.warning("Please upload a CSV file to begin.")

# 📤 Optional: Download log button
if os.path.exists("chat_log.txt"):
    with open("chat_log.txt", "r") as log_file:
        st.download_button("\U0001F4C4 Download Interaction Log", log_file, file_name="chat_log.txt")
