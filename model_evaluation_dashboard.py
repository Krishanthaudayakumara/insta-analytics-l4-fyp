"""
Model Evaluation Dashboard for Instagram User Behavior Analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

def load_evaluation_results(target_type="engagement"):
    """
    Load evaluation results from JSON file.
    
    Args:
        target_type: Type of target (engagement, likes, comments)
    
    Returns:
        dict: Latest evaluation results or empty dict if not found
    """
    filename = f'outputs/model_evaluation_{target_type}.json'
    
    try:
        with open(filename, 'r') as f:
            all_results = json.load(f)
        
        if all_results:
            # Return the most recent evaluation
            return all_results[-1]['evaluation_results']
        else:
            return []
    except FileNotFoundError:
        return []

def create_evaluation_comparison_df(target_types=["engagement", "likes", "comments"]):
    """
    Create a comprehensive comparison DataFrame of all model evaluations.
    
    Args:
        target_types: List of target types to include
    
    Returns:
        pandas.DataFrame: Comparison table of all models
    """
    all_results = []
    
    for target_type in target_types:
        results = load_evaluation_results(target_type)
        if results:
            for result in results:
                result['Target'] = target_type.title()
                all_results.append(result)
    
    if all_results:
        comparison_df = pd.DataFrame(all_results)
        
        # Reorder columns for better display
        key_columns = ['Target', 'Model', 'MSE', 'RMSE', 'MAE', 'R²_Score', 'MAPE_%', 
                      'CV_R²_Mean', 'CV_R²_Std', 'Within_10%_Error', 'Within_20%_Error']
        
        # Only include columns that exist
        available_columns = [col for col in key_columns if col in comparison_df.columns]
        other_columns = [col for col in comparison_df.columns if col not in available_columns]
        
        comparison_df = comparison_df[available_columns + other_columns]
        
        return comparison_df
    else:
        return pd.DataFrame()

def show_model_evaluation_dashboard():
    """
    Display the comprehensive model evaluation dashboard
    """
    st.header("🎯 Model Performance Evaluation Dashboard")
    
    # Action buttons section
    st.subheader("🚀 Model Training & Evaluation Actions")
    
    # Check if data is available in session state or try to load default data
    if 'df' not in st.session_state:
        try:
            df = pd.read_csv('data/processed_data/cleaned_merged_user_post_data.csv')
            st.session_state.df = df
            st.info("✅ Data loaded successfully for model training")
        except Exception as e:
            df = None
            st.warning(f"⚠️ No data available for training. Please load data first. Error: {e}")
    else:
        df = st.session_state.df
    
    # Action buttons in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎯 Train Engagement Models", 
                     help="Train and evaluate engagement prediction models (Linear Regression, Ridge, Random Forest)",
                     disabled=(df is None)):
            if df is not None:
                with st.spinner("Training engagement prediction models..."):
                    try:
                        from analysis.engagement_prediction import run
                        run(df)
                        st.success("✅ Engagement models trained and evaluated!")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error training engagement models: {e}")
    
    with col2:
        if st.button("👍💬 Train Likes/Comments Models", 
                     help="Train and evaluate separate models for likes and comments prediction",
                     disabled=(df is None)):
            if df is not None:
                with st.spinner("Training likes and comments prediction models..."):
                    try:
                        from analysis.engagement_prediction import train_and_save_like_comment_models
                        train_and_save_like_comment_models(df)
                        st.success("✅ Likes/Comments models trained and evaluated!")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error training likes/comments models: {e}")
    
    with col3:
        if st.button("🔄 Retrain All Models", 
                     help="Train all models (engagement, likes, comments) in sequence",
                     disabled=(df is None)):
            if df is not None:
                with st.spinner("Training all models..."):
                    try:
                        from analysis.engagement_prediction import run, train_and_save_like_comment_models
                        
                        # Train engagement models
                        st.info("Training engagement models...")
                        run(df)
                        
                        # Train likes/comments models
                        st.info("Training likes/comments models...")
                        train_and_save_like_comment_models(df)
                        
                        st.success("✅ All models trained and evaluated successfully!")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error training models: {e}")
    
    with col4:
        if st.button("🗑️ Clear All Evaluations", 
                     help="Remove all stored evaluation results and start fresh"):
            try:
                import os
                target_types = ["engagement", "likes", "comments"]
                cleared_count = 0
                
                for target_type in target_types:
                    filename = f'outputs/model_evaluation_{target_type}.json'
                    if os.path.exists(filename):
                        os.remove(filename)
                        cleared_count += 1
                
                if cleared_count > 0:
                    st.success(f"✅ Cleared {cleared_count} evaluation files")
                    st.rerun()
                else:
                    st.info("No evaluation files found to clear")
            except Exception as e:
                st.error(f"❌ Error clearing evaluations: {e}")
    
    st.markdown("---")
    
    # Quick Status Section
    st.subheader("📋 Current Model Status")
    
    # Check status of each model type
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        engagement_results = load_evaluation_results("engagement")
        if engagement_results:
            best_engagement = min(engagement_results, key=lambda x: x.get('MSE', float('inf')))
            st.metric(
                label="🎯 Engagement Models",
                value=f"{len(engagement_results)} trained",
                delta=f"Best R²: {best_engagement.get('R²_Score', 'N/A'):.3f}"
            )
            st.caption(f"Best: {best_engagement['Model']}")
        else:
            st.metric(
                label="🎯 Engagement Models", 
                value="Not trained",
                delta="Click 'Train Engagement Models' above"
            )
    
    with status_col2:
        likes_results = load_evaluation_results("likes")
        if likes_results:
            best_likes = min(likes_results, key=lambda x: x.get('MSE', float('inf')))
            st.metric(
                label="👍 Likes Models",
                value=f"{len(likes_results)} trained",
                delta=f"Best R²: {best_likes.get('R²_Score', 'N/A'):.3f}"
            )
            st.caption(f"Best: {best_likes['Model']}")
        else:
            st.metric(
                label="👍 Likes Models",
                value="Not trained", 
                delta="Click 'Train Likes/Comments Models' above"
            )
    
    with status_col3:
        comments_results = load_evaluation_results("comments")
        if comments_results:
            best_comments = min(comments_results, key=lambda x: x.get('MSE', float('inf')))
            st.metric(
                label="💬 Comments Models",
                value=f"{len(comments_results)} trained",
                delta=f"Best R²: {best_comments.get('R²_Score', 'N/A'):.3f}"
            )
            st.caption(f"Best: {best_comments['Model']}")
        else:
            st.metric(
                label="💬 Comments Models",
                value="Not trained",
                delta="Click 'Train Likes/Comments Models' above"
            )
    
    # Progress indicator
    total_possible = 9  # 3 models × 3 targets
    total_trained = len(engagement_results) + len(likes_results) + len(comments_results)
    progress = total_trained / total_possible if total_possible > 0 else 0
    
    st.progress(progress)
    st.caption(f"Model Training Progress: {total_trained}/{total_possible} models trained ({progress:.1%})")
    
    # Training recommendations
    if total_trained == 0:
        st.info("🚀 **Get Started:** Train your first models using the buttons above! We recommend starting with 'Retrain All Models' for comprehensive evaluation.")
    elif total_trained < total_possible:
        missing_targets = []
        if not engagement_results:
            missing_targets.append("Engagement")
        if not likes_results:
            missing_targets.append("Likes")
        if not comments_results:
            missing_targets.append("Comments")
        
        if missing_targets:
            st.warning(f"⚠️ **Missing Models:** {', '.join(missing_targets)} models not trained yet. Consider training them for complete evaluation coverage.")
    else:
        st.success("🎉 **All models trained!** Explore the comprehensive evaluation results in the tabs below.")
    
    # Show last training activity
    if total_trained > 0:
        try:
            import os
            from datetime import datetime
            
            latest_time = None
            latest_target = None
            
            for target in ["engagement", "likes", "comments"]:
                filename = f'outputs/model_evaluation_{target}.json'
                if os.path.exists(filename):
                    import json
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    if data:
                        timestamp_str = data[-1]['timestamp']
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        if latest_time is None or timestamp > latest_time:
                            latest_time = timestamp
                            latest_target = target
            
            if latest_time:
                time_diff = datetime.now() - latest_time
                if time_diff.total_seconds() < 3600:  # Less than 1 hour
                    st.caption(f"🕒 Last training: {latest_target} models ({time_diff.total_seconds()/60:.0f} minutes ago)")
                else:
                    st.caption(f"🕒 Last training: {latest_target} models ({latest_time.strftime('%Y-%m-%d %H:%M')})")
        except Exception:
            pass  # Silently ignore errors in timestamp display
    
    st.markdown("---")
    
    # Create tabs for different evaluation views
    eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs([
        "📊 Overview", "🔍 Detailed Metrics", "📈 Performance Comparison", "📋 Evaluation History"
    ])
    
    with eval_tab1:
        st.subheader("Model Performance Overview")
        
        # Quick overview cards
        col1, col2, col3 = st.columns(3)
        
        # Load latest results for each target type
        engagement_results = load_evaluation_results("engagement")
        likes_results = load_evaluation_results("likes")
        comments_results = load_evaluation_results("comments")
        
        with col1:
            st.metric(
                label="📈 Engagement Models",
                value=f"{len(engagement_results)} trained" if engagement_results else "0 trained",
                help="Number of trained engagement prediction models"
            )
            if engagement_results:
                best_model = min(engagement_results, key=lambda x: x.get('MSE', float('inf')))
                st.success(f"Best: {best_model['Model']} (R² = {best_model.get('R²_Score', 'N/A')})")
        
        with col2:
            st.metric(
                label="👍 Likes Models",
                value=f"{len(likes_results)} trained" if likes_results else "0 trained",
                help="Number of trained likes prediction models"
            )
            if likes_results:
                best_model = min(likes_results, key=lambda x: x.get('MSE', float('inf')))
                st.success(f"Best: {best_model['Model']} (R² = {best_model.get('R²_Score', 'N/A')})")
        
        with col3:
            st.metric(
                label="💬 Comments Models",
                value=f"{len(comments_results)} trained" if comments_results else "0 trained",
                help="Number of trained comments prediction models"
            )
            if comments_results:
                best_model = min(comments_results, key=lambda x: x.get('MSE', float('inf')))
                st.success(f"Best: {best_model['Model']} (R² = {best_model.get('R²_Score', 'N/A')})")
        
        # Performance Summary
        if engagement_results or likes_results or comments_results:
            st.markdown("---")
            st.subheader("Performance Summary")
            
            all_results = engagement_results + likes_results + comments_results
            if all_results:
                avg_r2 = np.mean([r.get('R²_Score', 0) for r in all_results])
                avg_mape = np.mean([r.get('MAPE_%', 0) for r in all_results])
                avg_within_20 = np.mean([r.get('Within_20%_Error', 0) for r in all_results])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average R² Score", f"{avg_r2:.3f}")
                with col2:
                    st.metric("Average MAPE", f"{avg_mape:.1f}%")
                with col3:
                    st.metric("Avg. Predictions within 20%", f"{avg_within_20:.1f}%")
        else:
            st.info("No model evaluation results found. Train models to see performance metrics.")
    
    with eval_tab2:
        st.subheader("Detailed Model Metrics")
        
        # Create comprehensive comparison table
        comparison_df = create_evaluation_comparison_df()
        
        if not comparison_df.empty:
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                target_filter = st.selectbox(
                    "Filter by Target:",
                    ["All"] + comparison_df['Target'].unique().tolist()
                )
            with col2:
                model_filter = st.selectbox(
                    "Filter by Model:",
                    ["All"] + comparison_df['Model'].str.replace(r'\s*\([^)]*\)', '', regex=True).unique().tolist()
                )
            
            # Apply filters
            filtered_df = comparison_df.copy()
            if target_filter != "All":
                filtered_df = filtered_df[filtered_df['Target'] == target_filter]
            if model_filter != "All":
                filtered_df = filtered_df[filtered_df['Model'].str.contains(model_filter, case=False)]
            
            # Display table with formatting
            if not filtered_df.empty:
                # Format numerical columns for better display
                display_df = filtered_df.copy()
                numeric_cols = ['MSE', 'RMSE', 'MAE', 'R²_Score', 'MAPE_%', 'CV_R²_Mean', 'CV_R²_Std']
                
                def format_value(x, is_percentage=False):
                    try:
                        if x is None or np.isnan(float(x)):
                            return "N/A"
                        if is_percentage:
                            return f"{float(x):.1f}%"
                        else:
                            return f"{float(x):.4f}"
                    except (ValueError, TypeError):
                        return "N/A"
                
                for col in numeric_cols:
                    if col in display_df.columns:
                        if col.endswith('_%'):
                            display_df[col] = display_df[col].apply(lambda x: format_value(x, True))
                        else:
                            display_df[col] = display_df[col].apply(lambda x: format_value(x, False))
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button for the comparison table
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Evaluation Results (CSV)",
                    data=csv,
                    file_name=f"model_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No results match the selected filters.")
        else:
            st.info("No detailed evaluation results available. Train models to see comprehensive metrics.")
    
    with eval_tab3:
        st.subheader("Performance Comparison Charts")
        
        comparison_df = create_evaluation_comparison_df()
        
        if not comparison_df.empty and len(comparison_df) > 1:
            # R² Score Comparison
            fig_r2 = px.bar(
                comparison_df,
                x='Model',
                y='R²_Score',
                color='Target',
                title="R² Score Comparison Across Models",
                labels={'R²_Score': 'R² Score', 'Model': 'Model Type'},
                height=400
            )
            fig_r2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_r2, use_container_width=True)
            
            # MAPE Comparison (lower is better)
            if 'MAPE_%' in comparison_df.columns:
                fig_mape = px.bar(
                    comparison_df,
                    x='Model',
                    y='MAPE_%',
                    color='Target',
                    title="Mean Absolute Percentage Error (MAPE) - Lower is Better",
                    labels={'MAPE_%': 'MAPE (%)', 'Model': 'Model Type'},
                    height=400
                )
                fig_mape.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_mape, use_container_width=True)
            
            # Accuracy within Error Bounds
            if 'Within_20%_Error' in comparison_df.columns:
                fig_accuracy = px.bar(
                    comparison_df,
                    x='Model',
                    y='Within_20%_Error',
                    color='Target',
                    title="Predictions Within 20% Error - Higher is Better",
                    labels={'Within_20%_Error': 'Accuracy (%)', 'Model': 'Model Type'},
                    height=400
                )
                fig_accuracy.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_accuracy, use_container_width=True)
            
            # Cross-validation R² vs Test R² scatter plot
            if 'CV_R²_Mean' in comparison_df.columns:
                fig_cv = px.scatter(
                    comparison_df,
                    x='CV_R²_Mean',
                    y='R²_Score',
                    color='Target',
                    size='MSE',
                    hover_data=['Model', 'MAPE_%'] if 'MAPE_%' in comparison_df.columns else ['Model'],
                    title="Cross-Validation R² vs Test R² (Size = MSE)",
                    labels={'CV_R²_Mean': 'Cross-Validation R² (Mean)', 'R²_Score': 'Test R²'},
                    height=500
                )
                # Add diagonal line for reference
                min_val = min(comparison_df['CV_R²_Mean'].min(), comparison_df['R²_Score'].min())
                max_val = max(comparison_df['CV_R²_Mean'].max(), comparison_df['R²_Score'].max())
                fig_cv.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", dash="dash"),
                )
                st.plotly_chart(fig_cv, use_container_width=True)
                st.caption("Points close to the diagonal line indicate good model generalization (similar CV and test performance)")
        else:
            st.info("Need at least 2 model evaluations to show comparison charts.")
    
    with eval_tab4:
        st.subheader("Evaluation History")
        
        # Load all historical evaluations for each target type
        target_types = ["engagement", "likes", "comments"]
        
        for target_type in target_types:
            filename = f'outputs/model_evaluation_{target_type}.json'
            try:
                with open(filename, 'r') as f:
                    all_evaluations = json.load(f)
                
                if all_evaluations:
                    st.write(f"**{target_type.title()} Prediction Models History:**")
                    
                    for i, evaluation in enumerate(reversed(all_evaluations[-5:])):  # Show last 5 evaluations
                        with st.expander(f"Evaluation {len(all_evaluations)-i} - {evaluation['timestamp']}"):
                            eval_df = pd.DataFrame(evaluation['evaluation_results'])
                            
                            # Format for display
                            display_cols = ['Model', 'MSE', 'RMSE', 'R²_Score', 'MAPE_%', 'Within_20%_Error']
                            display_eval_df = eval_df[display_cols] if all(col in eval_df.columns for col in display_cols) else eval_df
                            
                            st.dataframe(display_eval_df, use_container_width=True, hide_index=True)
                    
                    st.markdown("---")
            
            except FileNotFoundError:
                continue
        
        # Cleanup old evaluation files button
        if st.button("🗑️ Clear Evaluation History", help="Remove all stored evaluation results"):
            for target_type in target_types:
                filename = f'outputs/model_evaluation_{target_type}.json'
                try:
                    os.remove(filename)
                    st.success(f"Cleared {target_type} evaluation history")
                except FileNotFoundError:
                    pass
            st.rerun()

# Main function to be called from app.py
def main():
    show_model_evaluation_dashboard()

if __name__ == "__main__":
    main()
