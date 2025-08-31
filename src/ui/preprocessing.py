"""
Preprocessing UI Component
Handles data preprocessing interface
"""

import streamlit as st
import pandas as pd
from .base import BaseUIComponent


class PreprocessingComponent(BaseUIComponent):
    """Data preprocessing component"""
    
    def show(self):
        """Show data preprocessing options"""
        st.markdown("### 📊 Data Preprocessing")
        
        data_source = st.radio(
            "Select Data Source",
            ("Upload CSV", "Process Clustered Data")
        )

        if data_source == "Upload CSV":
            self._handle_csv_upload()
        else:
            self._handle_clustered_data()

    def _handle_csv_upload(self):
        """Handle CSV upload and processing"""
        st.markdown("#### Upload Raw Instagram Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                
                st.markdown("#### Raw Data Preview")
                st.dataframe(raw_df.head())
                
                if st.button("🚀 Preprocess Data", type="primary"):
                    with st.spinner("⚙️ Processing data... This may take a moment."):
                        self.data_processor.fit(raw_df)
                        processed_df = self.data_processor.transform()
                        
                        # Save processed data
                        processed_df.to_csv("outputs/preprocessed_data.csv", index=False)
                        
                        st.success("✅ Data preprocessing complete!")
                        st.markdown("#### Processed Data Preview")
                        st.dataframe(processed_df.head())
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

    def _handle_clustered_data(self):
        """Handle clustered data processing"""
        st.markdown("#### Process Clustered Instagram Data")
        
        cluster_names = self.clustered_data_processor.get_cluster_names()
        if not cluster_names:
            st.warning("⚠️ No cluster directories found in `data/clustered_data`.")
            return

        selected_cluster = st.selectbox("Select a cluster to process", cluster_names)
        
        if st.button("⚙️ Process Cluster", type="primary"):
            with st.spinner(f"Processing {selected_cluster}... This might take a while."):
                try:
                    # Process the cluster data
                    raw_df = self.clustered_data_processor.process_cluster(selected_cluster)
                    
                    if raw_df.empty:
                        st.error(f"❌ No data found in cluster '{selected_cluster}'. Check the data format.")
                        return
                    
                    # Make it compatible with existing pipeline
                    processed_df = self.clustered_data_processor.create_compatible_dataframe(raw_df)
                    
                    # Store in session state
                    st.session_state['clustered_df'] = processed_df
                    st.session_state['selected_cluster'] = selected_cluster
                    
                    st.success(f"✅ Cluster '{selected_cluster}' processed successfully!")
                    st.markdown("#### Processed Cluster Data Preview")
                    st.dataframe(processed_df.head())
                    
                    # Show data statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Posts", len(processed_df))
                    with col2:
                        st.metric("Unique Users", processed_df['username'].nunique())
                    with col3:
                        st.metric("Total Comments", processed_df['comment_text'].notna().sum())
                        
                except Exception as e:
                    st.error(f"❌ Error processing cluster: {str(e)}")
                    st.info("This might be due to data format issues. Please check the .info files structure.")

        if 'clustered_df' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Generate and Save Intermediate CSV"):
                    output_path = f"outputs/{st.session_state['selected_cluster']}_preprocessed.csv"
                    st.session_state['clustered_df'].to_csv(output_path, index=False)
                    st.success(f"✅ Intermediate CSV saved to `{output_path}`")
                    st.info("You can now use the 'Upload CSV' option with this generated file.")
            
            with col2:
                if st.button("🚀 Continue with This Data"):
                    # Save as the main preprocessed data for the pipeline
                    st.session_state['clustered_df'].to_csv("outputs/preprocessed_data.csv", index=False)
                    st.success("✅ Data set as main dataset for the pipeline!")
                    st.info("You can now proceed to the next steps in the pipeline.")
