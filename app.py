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

# Set up OpenAI API key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="InsightForge BI Assistant", layout="wide")
st.title("📊 InsightForge – AI-Powered Business Intelligence Assistant")

uploaded_file = st.sidebar.file_uploader("Upload Sales Data CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data Loaded")

    # Prepare documents
    documents = []
    for _, row in df.iterrows():
        doc = {
            "date": pd.to_datetime(row['Date']).strftime('%Y-%m-%d'),
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

    # RAG setup
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    user_query = st.text_input("💬 Ask a business question:")

    if user_query:
        result = qa_chain.invoke({"query": user_query})
        st.subheader("🧠 AI Insight")
        st.write(result["result"])

        # Log the interaction
        with open("chat_log.txt", "a") as log_file:
            log_file.write(f"Time: {datetime.now().isoformat()}\n")
            log_file.write(f"User Query: {user_query}\n")
            log_file.write(f"AI Response: {result['result']}\n")
            log_file.write("-" * 50 + "\n")

        # 📊 Visual triggers
        q = user_query.lower()

        if "sales trend" in q:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.to_period('M').astype(str)
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
    st.warning("Please upload a CSV file to begin.")

# 📄 Download log button
if os.path.exists("chat_log.txt"):
    with open("chat_log.txt", "r") as log_file:
        st.download_button("📄 Download Interaction Log", log_file, file_name="chat_log.txt")
