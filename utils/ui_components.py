"""Enhanced UI components for better user experience."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import pandas as pd

def display_audio_info(audio_info: Dict[str, Any]):
    """Display audio file information in a nice format."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", audio_info.get('duration_formatted', 'Unknown'))
    
    with col2:
        st.metric("File Size", f"{audio_info.get('file_size_mb', 0):.1f} MB")
    
    with col3:
        st.metric("Sample Rate", f"{audio_info.get('sample_rate', 0)} Hz")
    
    with col4:
        st.metric("Est. Chunks", audio_info.get('estimated_chunks', 0))

def display_transcription_results(results: Dict[str, Any]):
    """Display transcription results with metrics."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Language", results.get('language', 'Unknown').upper())
    
    with col2:
        st.metric("Confidence", f"{results.get('confidence', 0):.1%}")
    
    with col3:
        st.metric("Chunks Processed", results.get('chunks_processed', 0))
    
    # Word count and reading time
    word_count = len(results.get('transcription', '').split())
    reading_time = max(1, word_count // 200)  # Assume 200 words per minute
    
    col4, col5 = st.columns(2)
    with col4:
        st.metric("Word Count", word_count)
    with col5:
        st.metric("Reading Time", f"{reading_time} min")

def display_keywords(keywords: List[Dict[str, Any]]):
    """Display keywords with interactive visualization."""
    if not keywords:
        st.warning("No keywords extracted.")
        return
    
    # Create DataFrame for better display
    df = pd.DataFrame(keywords)
    
    # Keywords table
    st.subheader("📊 Keyword Analysis")
    
    # Interactive bar chart
    fig = px.bar(
        df.head(10),
        x='score',
        y='keyword',
        orientation='h',
        color='score',
        color_continuous_scale='viridis',
        title="Top Keywords by Relevance Score"
    )
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Keywords by confidence
    confidence_counts = df['confidence'].value_counts()
    fig_pie = px.pie(
        values=confidence_counts.values,
        names=confidence_counts.index,
        title="Keywords by Confidence Level"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed table
    with st.expander("📋 Detailed Keywords Table"):
        st.dataframe(
            df,
            column_config={
                "keyword": "Keyword",
                "score": st.column_config.ProgressColumn(
                    "Relevance Score",
                    help="Higher scores indicate more relevant keywords",
                    min_value=0,
                    max_value=1,
                ),
                "confidence": "Confidence Level"
            },
            hide_index=True
        )

def display_summary(summary_data: Dict[str, Any]):
    """Display summary with enhanced formatting."""
    if not summary_data or not summary_data.get('summary'):
        st.warning("No summary generated.")
        return
    
    st.subheader("📋 Summary")
    
    # Summary metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Summary Method", summary_data.get('method', 'Unknown').title())
    with col2:
        st.metric("Chunks Processed", summary_data.get('chunks_processed', 0))
    
    # Main summary
    st.markdown("### 📝 Full Summary")
    st.write(summary_data['summary'])
    
    # Bullet points
    if summary_data.get('bullet_points'):
        st.markdown("### 🎯 Key Points")
        for i, point in enumerate(summary_data['bullet_points'], 1):
            st.markdown(f"{i}. {point}")

def display_topics(topics: List[Dict[str, Any]]):
    """Display topics with interactive visualization."""
    if not topics:
        st.warning("No topics detected.")
        return
    
    st.subheader("🧠 Topic Analysis")
    
    # Topics overview
    topic_df = pd.DataFrame([
        {
            'Topic': f"Topic {t['topic_id']}",
            'Top Words': ', '.join(t['words'][:3]),
            'Document Count': t['document_count'],
            'Relevance': sum(t['scores'][:3]) / 3
        }
        for t in topics
    ])
    
    # Topic distribution chart
    fig = px.bar(
        topic_df,
        x='Topic',
        y='Document Count',
        color='Relevance',
        title="Topic Distribution",
        color_continuous_scale='plasma'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed topic information
    for i, topic in enumerate(topics):
        with st.expander(f"🏷️ Topic {topic['topic_id']}: {', '.join(topic['words'][:3])}"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Words:**")
                word_df = pd.DataFrame({
                    'Word': topic['words'],
                    'Score': topic['scores']
                })
                st.dataframe(word_df, hide_index=True)
            
            with col2:
                st.markdown("**Representative Excerpts:**")
                for doc in topic.get('representative_docs', [])[:3]:
                    st.markdown(f"• {doc[:100]}...")

def display_export_options(transcription: str, keywords: List[Dict], summary: Dict, topics: List[Dict]):
    """Display export options for results."""
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Download Transcript"):
            st.download_button(
                label="Download as TXT",
                data=transcription,
                file_name="transcript.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("📊 Download Analysis"):
            import json
            analysis_data = {
                'keywords': keywords,
                'summary': summary,
                'topics': topics
            }
            st.download_button(
                label="Download as JSON",
                data=json.dumps(analysis_data, indent=2),
                file_name="analysis.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📋 Download Report"):
            report = generate_full_report(transcription, keywords, summary, topics)
            st.download_button(
                label="Download Full Report",
                data=report,
                file_name="podcast_analysis_report.md",
                mime="text/markdown"
            )

def generate_full_report(transcription: str, keywords: List[Dict], summary: Dict, topics: List[Dict]) -> str:
    """Generate a comprehensive markdown report."""
    report = f"""# Podcast Analysis Report

## 📝 Transcription
{transcription}

## 🔑 Keywords
"""
    
    for kw in keywords[:10]:
        report += f"- **{kw['keyword']}** (Score: {kw['score']:.3f}, Confidence: {kw['confidence']})\n"
    
    report += f"""
## 📋 Summary
{summary.get('summary', 'No summary available')}

### Key Points:
"""
    
    for i, point in enumerate(summary.get('bullet_points', []), 1):
        report += f"{i}. {point}\n"
    
    report += "\n## 🧠 Topics\n"
    
    for topic in topics:
        report += f"### Topic {topic['topic_id']}\n"
        report += f"**Keywords:** {', '.join(topic['words'])}\n"
        report += f"**Document Count:** {topic['document_count']}\n\n"
    
    return report

def show_processing_animation():
    """Show a processing animation."""
    return st.empty()