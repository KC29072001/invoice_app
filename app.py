import streamlit as st
from dotenv import load_dotenv
from utils import create_docs

def main():
    load_dotenv()
    st.set_page_config(page_title="Invoice Extraction Bot")
    st.title("ChatBot...💁 ")
    st.subheader("I can help you in extracting  data")
    
    # Upload the Invoices (pdf files)
    pdf_files = st.file_uploader("Upload invoices here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)
    
    submit = st.button("Extract Data")
    
    if submit:
        if pdf_files:
            with st.spinner('Wait for it...'):
                df = create_docs(pdf_files)
                st.write(df.head())
                data_as_csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download data as CSV", 
                    data_as_csv, 
                    "extracted_data.csv",
                    "text/csv",
                    key="download-csv",
                )
            st.success("Data extraction completed successfully!")
        else:
            st.warning("Please upload at least one PDF file.")

if __name__ == '__main__':
    main()
