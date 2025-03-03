"""
Streamlit application for the Cloud Architecture Recommendation System.

This application provides a web interface for users to interact with the
multi-agent system for cloud architecture recommendations.
"""
import streamlit as st
import asyncio
import json
import os
os.environ["PYTORCH_JIT"] = "0"  # Disable PyTorch JIT to avoid the module path error

import pandas as pd
from typing import Dict, Any, List, Optional
import datetime
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

from agents.master_agent import MasterAgent, FileType, UserRequirements
from agents.scraping_agent import ScrapingAgent
from agents.solution_architect_agent import SolutionArchitectAgent
from agents.mathematics_agent import MathematicsAgent

# Page configuration
st.set_page_config(
    page_title="Cloud Architecture Recommendation System",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for storing application state
if "master_agent" not in st.session_state:
    st.session_state.master_agent = None
if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = None
if "report_result" not in st.session_state:
    st.session_state.report_result = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "User Requirements"


async def initialize_agents() -> MasterAgent:
    """
    Initialize all agents in the system.
    
    Returns:
        Initialized master agent with all subagents.
    """
    # Initialize agents with appropriate configurations
    use_api = bool(os.getenv("DEEPSEEK_API_KEY", ""))
    
    # Create agents
    scraping_agent = ScrapingAgent()
    
    solution_architect_agent = SolutionArchitectAgent(
        use_api=use_api,
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-coder-v1.5-instruct")
    )
    
    mathematics_agent = MathematicsAgent()
    
    # Create master agent
    master_agent = MasterAgent(
        scraping_agent=scraping_agent,
        solution_architect_agent=solution_architect_agent,
        mathematics_agent=mathematics_agent
    )
    
    return master_agent


def display_welcome():
    """Display welcome message and application description."""
    st.title("Cloud Architecture Recommendation System")
    
    st.markdown("""
    This system helps you determine the most cost-effective and appropriate cloud 
    architecture based on your specific requirements. It uses a multi-agent approach 
    to analyze requirements, scrape pricing data, and generate recommendations.
    
    The system includes:
    - **Scraping Agent** that collects current pricing information from cloud providers
    - **Solution Architect Agent** that designs appropriate architectures
    - **Mathematics Agent** that validates costs and budget constraints
    - **Master Agent** that orchestrates the process and generates reports
    
    Get started by filling out your requirements below.
    """)


def collect_user_requirements() -> Dict[str, Any]:
    """
    Collect user requirements through the Streamlit interface.
    
    Returns:
        Dictionary of user requirements.
    """
    st.header("User Requirements")
    
    # Basic requirements
    col1, col2 = st.columns(2)
    
    with col1:
        expected_users = st.number_input(
            "Expected Users per Month",
            min_value=1,
            value=1000,
            help="Number of users expected to use the system monthly"
        )
    
    with col2:
        max_monthly_budget = st.number_input(
            "Maximum Monthly Budget (USD)",
            min_value=0.0,
            value=2000.0,
            help="Maximum budget for cloud resources per month"
        )
    
    # File types
    st.subheader("File Types")
    st.markdown("Specify the types of files you'll be storing and their characteristics.")
    
    file_types = {}
    
    # Predefined file type templates
    add_predefined = st.checkbox("Add predefined file types", value=False)
    if add_predefined:
        predefined_types = st.multiselect(
            "Select predefined file types",
            options=["PDF Documents", "Images", "Videos", "Structured Data (CSV/JSON)", "Cache Data"]
        )
        
        if "PDF Documents" in predefined_types:
            file_types["pdf"] = {
                "monthly_volume": 5000,
                "avg_size_mb": 2.5
            }
        
        if "Images" in predefined_types:
            file_types["images"] = {
                "monthly_volume": 10000,
                "avg_size_mb": 1.0
            }
        
        if "Videos" in predefined_types:
            file_types["videos"] = {
                "monthly_volume": 500,
                "avg_size_mb": 100.0
            }
        
        if "Structured Data (CSV/JSON)" in predefined_types:
            file_types["structured_data"] = {
                "monthly_volume": 20000,
                "avg_size_mb": 0.5
            }
        
        if "Cache Data" in predefined_types:
            file_types["cache"] = {
                "monthly_volume": 50000,
                "avg_size_mb": 0.1
            }
    
    # Custom file types
    num_custom_types = st.number_input("Number of custom file types", min_value=0, max_value=10, value=0)
    
    for i in range(num_custom_types):
        st.markdown(f"#### Custom File Type {i+1}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            type_name = st.text_input(f"Type Name #{i+1}", key=f"type_name_{i}")
        
        with col2:
            monthly_volume = st.number_input(
                f"Monthly Volume #{i+1}",
                min_value=1,
                value=1000,
                key=f"monthly_volume_{i}"
            )
        
        with col3:
            avg_size_mb = st.number_input(
                f"Average Size (MB) #{i+1}",
                min_value=0.01,
                value=1.0,
                key=f"avg_size_mb_{i}"
            )
        
        if type_name:
            file_types[type_name] = {
                "monthly_volume": monthly_volume,
                "avg_size_mb": avg_size_mb
            }
    
    # Cloud provider preferences
    st.subheader("Cloud Provider Preferences")
    
    preferred_providers = st.multiselect(
        "Preferred Cloud Providers",
        options=["aws", "gcp", "azure"],
        default=["aws", "gcp", "azure"],
        help="Select cloud providers in order of preference"
    )
    
    coupling_preference = st.radio(
        "Architecture Coupling Preference",
        options=["tight", "loose"],
        index=1,
        help="Tight coupling uses a single provider, loose coupling allows mixed providers"
    )
    
    # Workload characteristics
    st.subheader("Workload Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        access_pattern = st.selectbox(
            "Data Access Pattern",
            options=["read_heavy", "write_heavy", "balanced", "analytics"],
            index=0,
            help="Expected pattern of data access"
        )
    
    with col2:
        transaction_requirements = st.selectbox(
            "Transaction Requirements",
            options=["low", "medium", "high", "critical"],
            index=0,
            help="Importance of transaction consistency and reliability"
        )
    
    # Compliance and data residency
    st.subheader("Compliance and Data Residency")
    
    compliance = st.multiselect(
        "Compliance Requirements",
        options=["hipaa", "pci", "gdpr", "fedramp", "soc", "cmmc"],
        default=[],
        help="Select applicable compliance requirements"
    )
    
    data_residency = st.selectbox(
        "Data Residency Requirements",
        options=["none", "us", "eu", "asia", "australia"],
        index=0,
        help="Geographic restrictions for data storage"
    )
    
    # Growth projections
    st.subheader("Growth Projections")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        monthly_user_growth_rate = st.slider(
            "Monthly User Growth Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="Expected monthly percentage growth in users"
        ) / 100.0
    
    with col2:
        monthly_storage_growth_rate = st.slider(
            "Monthly Storage Growth Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=0.5,
            help="Expected monthly percentage growth in storage"
        ) / 100.0
    
    with col3:
        time_horizon_months = st.number_input(
            "Planning Time Horizon (Months)",
            min_value=1,
            max_value=60,
            value=12,
            help="Number of months to consider for planning"
        )
    
    # Additional requirements
    st.subheader("Additional Requirements")
    
    serverless_preference = st.checkbox(
        "Preference for Serverless Architecture",
        value=False,
        help="Indicate if you prefer serverless components where applicable"
    )
    
    additional_requirements = st.multiselect(
        "Additional Requirements",
        options=["strong_consistency", "analytics", "high_availability", "disaster_recovery"],
        default=[],
        help="Select any additional requirements for your architecture"
    )
    
    # Compile all requirements
    requirements = {
        "expected_users": expected_users,
        "max_monthly_budget": max_monthly_budget,
        "file_types": file_types,
        "preferred_providers": preferred_providers,
        "coupling_preference": coupling_preference,
        "access_pattern": access_pattern,
        "transaction_requirements": transaction_requirements,
        "compliance": compliance,
        "data_residency": data_residency,
        "monthly_user_growth_rate": monthly_user_growth_rate,
        "monthly_storage_growth_rate": monthly_storage_growth_rate,
        "time_horizon_months": time_horizon_months,
        "serverless_preference": serverless_preference,
        "requirements": additional_requirements
    }
    
    return requirements


def display_recommendation(recommendation: Dict[str, Any]):
    """
    Display the architecture recommendation.
    
    Args:
        recommendation: The recommendation dictionary to display.
    """
    if not recommendation:
        st.warning("No recommendation available.")
        return
    
    st.header("Recommended Architecture")
    
    # Recommendation summary
    summary = recommendation.get("recommendation_summary", {})
    
    cols = st.columns(4)
    with cols[0]:
        st.metric("Architecture", summary.get("architecture_name", ""))
    with cols[1]:
        st.metric("Provider", summary.get("provider", "").upper())
    with cols[2]:
        st.metric("Monthly Cost", f"${summary.get('monthly_cost', 0):.2f}")
    with cols[3]:
        budget_met = summary.get("budget_met", False)
        st.metric(
            "Budget Status", 
            "Within Budget" if budget_met else "Over Budget",
            delta="✓" if budget_met else "✗"
        )
    
    # Architecture explanation
    st.subheader("Architecture Overview")
    st.markdown(recommendation.get("explanation", "No explanation available."))
    
    # Components
    st.subheader("Architecture Components")
    
    components = recommendation.get("proposed_architecture", {}).get("components", {})
    if components:
        component_data = []
        for name, details in components.items():
            component_data.append({
                "Component": name.replace("_", " ").title(),
                "Provider": details.get("provider", "").upper(),
                "Service": details.get("service_type", ""),
                "Tier": details.get("tier", ""),
                "Description": details.get("description", "")
            })
        
        st.dataframe(pd.DataFrame(component_data))
    else:
        st.info("No component details available.")
    
    # Cost breakdown
    st.subheader("Cost Breakdown")
    
    cost_analysis = recommendation.get("cost_analysis", {})
    if cost_analysis:
        cost_data = []
        for component in cost_analysis.get("cost_components", []):
            cost_data.append({
                "Component": component.get("name", ""),
                "Monthly Cost": f"${component.get('monthly_cost', 0):.2f}",
                "Yearly Cost": f"${component.get('yearly_cost', 0):.2f}",
                "Description": component.get("description", ""),
                "Scaling Factor": component.get("scaling_factor", "")
            })
        
        if cost_data:
            st.dataframe(pd.DataFrame(cost_data))
        
        # Cost recommendations
        if "cost_recommendations" in cost_analysis and cost_analysis["cost_recommendations"]:
            st.subheader("Cost Optimization Recommendations")
            for i, rec in enumerate(cost_analysis["cost_recommendations"]):
                st.markdown(f"{i+1}. {rec}")
    else:
        st.info("No cost analysis available.")
    
    # Alternatives
    if "alternatives" in recommendation and recommendation["alternatives"]:
        st.subheader("Alternative Architectures")
        
        for i, alt in enumerate(recommendation["alternatives"]):
            with st.expander(f"Alternative {i+1}: {alt.get('name', '')}"):
                st.markdown(f"**Provider:** {alt.get('provider', '').upper()}")
                st.markdown(f"**Pattern:** {alt.get('pattern', '').replace('_', ' ').title()}")
                st.markdown(f"**Suitability Score:** {alt.get('suitability_score', 0):.2f}")
                
                if "components" in alt and alt["components"]:
                    st.markdown("**Components:**")
                    for comp_name, comp_details in alt["components"].items():
                        st.markdown(f"- **{comp_name.replace('_', ' ').title()}**: {comp_details.get('description', '')} ({comp_details.get('provider', '').upper()} {comp_details.get('service_type', '')})")


def display_report(report: Dict[str, Any]):
    """
    Display the detailed report.
    
    Args:
        report: The report dictionary to display.
    """
    if not report:
        st.warning("No report available.")
        return
    
    st.header("Detailed Architecture Report")
    
    # Executive Summary
    st.subheader("Executive Summary")
    st.markdown(report.get("summary", "No summary available."))
    
    # Implementation Steps
    if "implementation_steps" in report:
        st.subheader("Implementation Steps")
        
        for i, step in enumerate(report["implementation_steps"]):
            with st.expander(f"Step {i+1}: {step.get('title', '')}"):
                st.markdown(step.get("description", ""))
                
                if "sub_steps" in step and step["sub_steps"]:
                    st.markdown("**Detailed Steps:**")
                    for j, sub_step in enumerate(step["sub_steps"]):
                        st.markdown(f"{j+1}. {sub_step}")
    
    # Cost Analysis
    if "cost_analysis" in report:
        st.subheader("Detailed Cost Analysis")
        
        cost = report["cost_analysis"]
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Monthly Total", f"${cost.get('summary', {}).get('monthly_total', 0):.2f}")
        with cols[1]:
            st.metric("Yearly Total", f"${cost.get('summary', {}).get('yearly_total', 0):.2f}")
        with cols[2]:
            st.metric("3-Year Total", f"${cost.get('summary', {}).get('three_year_total', 0):.2f}")
        with cols[3]:
            st.metric("5-Year Total", f"${cost.get('summary', {}).get('five_year_total', 0):.2f}")
        
        # Growth Projections
        if "growth_projections" in cost and cost["growth_projections"].get("monthly_costs") and cost["growth_projections"].get("months"):
            st.subheader("Cost Growth Projections")
            
            growth = cost["growth_projections"]
            chart_data = pd.DataFrame({
                "Month": growth["months"],
                "Monthly Cost": growth["monthly_costs"],
                "Cumulative Cost": growth.get("cumulative_costs", [0] * len(growth["months"]))
            })
            
            st.line_chart(chart_data, x="Month", y=["Monthly Cost", "Cumulative Cost"])
        
        # Sensitivity Analysis
        if "sensitivity_analysis" in cost:
            st.subheader("Cost Sensitivity Analysis")
            
            sensitivity = cost["sensitivity_analysis"]
            
            if "top_cost_drivers" in sensitivity:
                st.markdown("**Top Cost Drivers:**")
                
                driver_data = []
                for driver in sensitivity["top_cost_drivers"]:
                    driver_data.append({
                        "Component": driver.get("component", ""),
                        "Monthly Cost": f"${driver.get('monthly_cost', 0):.2f}",
                        "Impact of 20% Increase": f"${driver.get('increase_impact', 0):.2f}",
                        "Percentage Impact": f"{driver.get('increase_percentage', 0):.2f}%"
                    })
                
                if driver_data:
                    st.dataframe(pd.DataFrame(driver_data))
            
            if "cost_per_user" in sensitivity and sensitivity["cost_per_user"]:
                st.metric("Cost Per User", f"${sensitivity['cost_per_user']:.2f}")
            
            if "breakeven_users" in sensitivity and sensitivity["breakeven_users"]:
                st.metric("Breakeven Users", sensitivity["breakeven_users"])
    
    # Architecture Diagrams
    if "architecture_diagrams" in report:
        st.subheader("Architecture Diagrams")
        
        for name, diagram in report["architecture_diagrams"].items():
            with st.expander(f"{name.replace('_', ' ').title()} Diagram"):
                st.markdown(diagram)
    
    # Appendices
    if "appendices" in report:
        st.subheader("Appendices")
        
        appendices = report["appendices"]
        
        # Pricing Data Summary
        if "pricing_data_summary" in appendices:
            with st.expander("Pricing Data Summary"):
                pricing = appendices["pricing_data_summary"]
                
                if "summary" in pricing:
                    st.markdown(f"**Data scraped at:** {pricing['summary'].get('scraped_at', 'Unknown')}")
                    st.markdown(f"**Providers:** {', '.join(pricing['summary'].get('providers', []))}")
                
                if "services_by_category" in pricing:
                    categories = pricing["services_by_category"]
                    
                    for category, services in categories.items():
                        if services:
                            st.markdown(f"**{category.replace('_', ' ').title()} Services:** {', '.join(services)}")


async def run_recommendation(requirements: Dict[str, Any]):
    """
    Run the recommendation workflow using the master agent.
    
    Args:
        requirements: User requirements dictionary.
    """
    # Initialize master agent if not already initialized
    if not st.session_state.master_agent:
        with st.spinner("Initializing agents..."):
            st.session_state.master_agent = await initialize_agents()
    
    # Run the recommendation
    with st.spinner("Generating architecture recommendation... This may take a few minutes."):
        result = await st.session_state.master_agent.run(requirements)
    
    # Store result
    if "error" in result:
        st.error(f"Error generating recommendation: {result['error']}")
        st.session_state.recommendation_result = None
        return False
    else:
        st.session_state.recommendation_result = result.get("recommendation", {})
        return True


async def run_report_generation():
    """Generate a detailed report using the master agent."""
    if not st.session_state.master_agent:
        st.error("Master agent not initialized. Please generate a recommendation first.")
        return False
    
    # Run the report generation
    with st.spinner("Generating detailed report... This may take a few minutes."):
        result = await st.session_state.master_agent.run({"command": "generate_report"})
    
    # Store result
    if "error" in result:
        st.error(f"Error generating report: {result['error']}")
        st.session_state.report_result = None
        return False
    else:
        st.session_state.report_result = result.get("report", {})
        return True


def check_api_key():
    """Check if the DeepSeek API key is configured."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.sidebar.warning(
            "⚠️ DeepSeek API key not found. Set DEEPSEEK_API_KEY in your .env file for enhanced recommendations. "
            "The system will run with basic functionality."
        )
    else:
        st.sidebar.success("✅ DeepSeek API key configured")


def change_page(page_name: str):
    """
    Change the current page.
    
    Args:
        page_name: Name of the page to navigate to.
    """
    st.session_state.current_page = page_name


def main():
    """Main application entry point."""
    # Display welcome message
    display_welcome()
    
    # Check API key
    check_api_key()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        options=["User Requirements", "Recommendation", "Detailed Report"],
        index=["User Requirements", "Recommendation", "Detailed Report"].index(st.session_state.current_page)
    )
    
    # Update current page in session state
    st.session_state.current_page = page
    
    # Page routing
    if page == "User Requirements":
        st.write("---")
        requirements = collect_user_requirements()
        
        if st.button("Generate Recommendation"):
            st.session_state.is_processing = True
            success = asyncio.run(run_recommendation(requirements))
            st.session_state.is_processing = False
            
            if success:
                st.success("Recommendation generated successfully!")
                # Navigate to recommendation page
                change_page("Recommendation")
                st.rerun()
    
    elif page == "Recommendation":
        st.write("---")
        if st.session_state.recommendation_result:
            display_recommendation(st.session_state.recommendation_result)
            
            if st.button("Generate Detailed Report"):
                st.session_state.is_processing = True
                success = asyncio.run(run_report_generation())
                st.session_state.is_processing = False
                
                if success:
                    st.success("Report generated successfully!")
                    # Navigate to report page
                    change_page("Detailed Report")
                    st.rerun()
        else:
            st.info("No recommendation available. Please go to User Requirements and generate a recommendation first.")
    
    elif page == "Detailed Report":
        st.write("---")
        if st.session_state.report_result:
            display_report(st.session_state.report_result)
            
            # Export report option
            if st.download_button(
                label="Export Report as JSON",
                data=json.dumps(st.session_state.report_result, indent=2),
                file_name=f"cloud_architecture_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            ):
                st.success("Report downloaded successfully!")
        else:
            st.info("No report available. Please generate a recommendation and then a report.")
    
    # Footer
    st.write("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 0.8em;">
        Cloud Architecture Recommendation System | Created with Streamlit | Using Open-Source Tools
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()