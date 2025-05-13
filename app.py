# 📦 Imports
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

# 🔐 Secure API Key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ⚙️ Page setup
st.set_page_config(page_title="InsightForge BI Assistant", layout="wide")
st.title("📊 InsightForge – AI-Powered Business Intelligence Assistant")

# 📁 Upload CSV file
uploaded_file = st.sidebar.file_uploader("Upload Sales Data CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data Loaded")

    # 📚 Prepare records for embedding
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

    # ➕ Add summaries for better context retrieval
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M').astype(str)

    for month, total in df.groupby('Month')['Sales'].sum().items():
        documents.append(Document(page_content=f"Monthly Summary - {month}: Total Sales = {total}"))

    for _, row in df.groupby(['Product', 'Region'])['Sales'].sum().reset_index().iterrows():
        documents.append(Document(page_content=f"Product: {row['Product']}, Region: {row['Region']}, Total Sales: {row['Sales']}"))

    for _, row in df.groupby('Product')['Customer_Satisfaction'].mean().reset_index().iterrows():
        documents.append(Document(page_content=f"Product: {row['Product']}, Avg Satisfaction: {round(row['Customer_Satisfaction'], 2)}"))

    # 🔗 RAG System Setup
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # 💬 Prompt input
    user_query = st.text_input("💬 Ask a business question:")

    if user_query:
        result = qa_chain.invoke({"query": user_query})
        st.subheader("🧠 AI Insight")
        st.write(result["result"])

        # 🪵 Log interaction
        with open("chat_log.txt", "a") as log_file:
            log_file.write(f"Time: {datetime.now().isoformat()}\n")
            log_file.write(f"User Query: {user_query}\n")
            log_file.write(f"AI Response: {result['result']}\n")
            log_file.write("-" * 50 + "\n")

        # 🎯 Essential Visuals Triggered by Smart Keywords
        q = user_query.lower()

        # 📈 Show sales trend if prompt involves time/seasonality
        if "sales trend" in q or "monthly sales" in q or "over time" in q:
            monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=monthly_sales, x='Month', y='Sales', marker='o', ax=ax)
            ax.set_title('Monthly Sales Trend')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        # 📊 Show bar chart of sales by product
        elif "sales by product" in q or "compare products" in q or "product performance" in q:
            fig, ax = plt.subplots()
            sns.barplot(data=df, x='Product', y='Sales', estimator=sum, ax=ax)
            ax.set_title('Total Sales by Product')
            st.pyplot(fig)

        # 😊 Show satisfaction distribution if query involves satisfaction scores
        elif "satisfaction distribution" in q or "customer satisfaction" in q:
            fig, ax = plt.subplots()
            sns.histplot(df['Customer_Satisfaction'], bins=20, kde=True, ax=ax)
            ax.set_title('Customer Satisfaction Distribution')
            st.pyplot(fig)

# ⚠️ Reminder if file not uploaded
else:
    st.warning("Please upload a CSV file to begin.")

# 📄 Download logged interaction history
if os.path.exists("chat_log.txt"):
    with open("chat_log.txt", "r") as log_file:
        st.download_button("📄 Download Interaction Log", log_file, file_name="chat_log.txt")
