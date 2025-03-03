#!/usr/bin/env python3
"""
Main entry point for the Cloud Architecture Recommendation System.

This script initializes the multi-agent system and provides a command-line interface
for users to interact with the system.
"""
import asyncio
import argparse
import json
import os
import sys
from typing import Dict, Any, List, Optional
import pandas as pd
from tabulate import tabulate
import markdown
import datetime

from agents.master_agent import MasterAgent
from agents.scraping_agent import ScrapingAgent
from agents.solution_architect_agent import SolutionArchitectAgent
from agents.mathematics_agent import MathematicsAgent


class CloudArchitectApp:
    """Main application class for the Cloud Architecture Recommendation System."""
    
    def __init__(self):
        """Initialize the application."""
        self.master_agent = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the multi-agent system."""
        if not self.initialized:
            print("Initializing Cloud Architecture Recommendation System...")
            scraping_agent = ScrapingAgent()
            solution_architect_agent = SolutionArchitectAgent()
            mathematics_agent = MathematicsAgent()
            
            self.master_agent = MasterAgent(
                scraping_agent=scraping_agent,
                solution_architect_agent=solution_architect_agent,
                mathematics_agent=mathematics_agent
            )
            self.initialized = True
            print("System initialized successfully.")
    
    async def run_recommendation(self, requirements: Dict[str, Any]):
        """
        Generate architecture recommendations based on user requirements.
        
        Args:
            requirements: User requirements dictionary.
            
        Returns:
            Recommendation results as a dictionary.
        """
        if not self.initialized:
            await self.initialize()
        
        print("Generating architecture recommendations...")
        result = await self.master_agent.run(requirements)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return None
        
        return result
    
    async def generate_report(self):
        """
        Generate a comprehensive report based on the latest recommendation.
        
        Returns:
            Report data dictionary.
        """
        if not self.initialized:
            await self.initialize()
        
        print("Generating detailed report...")
        result = await self.master_agent.run({"command": "generate_report"})
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return None
        
        return result.get("report")
    
    def format_recommendation_as_markdown(self, recommendation: Dict[str, Any]) -> str:
        """
        Format the recommendation results as markdown for better readability.
        
        Args:
            recommendation: The recommendation dictionary from the master agent.
            
        Returns:
            Formatted markdown string.
        """
        if not recommendation:
            return "# No recommendation available."
        
        markdown_output = []
        
        # Add title and timestamp
        markdown_output.append("# Cloud Architecture Recommendation")
        markdown_output.append(f"*Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        markdown_output.append("")
        
        # Add recommendation summary
        markdown_output.append("## Recommendation Summary")
        summary = recommendation.get("recommendation_summary", {})
        
        summary_table = [
            ["Architecture", summary.get("architecture_name", "N/A")],
            ["Provider", summary.get("provider", "N/A").upper()],
            ["Monthly Cost", f"${summary.get('monthly_cost', 0):.2f}"],
            ["Yearly Cost", f"${summary.get('yearly_cost', 0):.2f}"],
            ["Budget Met", "✅ Yes" if summary.get("budget_met", False) else "❌ No"],
            ["Suitability Score", f"{summary.get('suitability_score', 0):.2f}/1.00"]
        ]
        
        markdown_output.append(tabulate(summary_table, tablefmt="pipe"))
        markdown_output.append("")
        
        # Add architecture explanation
        markdown_output.append("## Architecture Details")
        markdown_output.append(recommendation.get("explanation", "No explanation available."))
        markdown_output.append("")
        
        # Add components table
        markdown_output.append("## Architecture Components")
        
        components = recommendation.get("proposed_architecture", {}).get("components", {})
        if components:
            component_data = []
            headers = ["Component", "Provider", "Service", "Tier", "Description"]
            
            for name, details in components.items():
                component_data.append([
                    name.replace("_", " ").title(),
                    details.get("provider", "").upper(),
                    details.get("service_type", ""),
                    details.get("tier", ""),
                    details.get("description", "")
                ])
            
            markdown_output.append(tabulate(component_data, headers=headers, tablefmt="pipe"))
        else:
            markdown_output.append("No component details available.")
        
        markdown_output.append("")
        
        # Add cost analysis
        markdown_output.append("## Cost Analysis")
        
        cost_analysis = recommendation.get("cost_analysis", {})
        if cost_analysis:
            cost_headers = ["Component", "Monthly Cost", "Yearly Cost", "Description"]
            cost_data = []
            
            for component in cost_analysis.get("cost_components", []):
                cost_data.append([
                    component.get("name", ""),
                    f"${component.get('monthly_cost', 0):.2f}",
                    f"${component.get('yearly_cost', 0):.2f}",
                    component.get("description", "")
                ])
            
            if cost_data:
                markdown_output.append(tabulate(cost_data, headers=cost_headers, tablefmt="pipe"))
            
            # Add cost recommendations
            if "cost_recommendations" in cost_analysis and cost_analysis["cost_recommendations"]:
                markdown_output.append("")
                markdown_output.append("### Cost Optimization Recommendations")
                
                for i, rec in enumerate(cost_analysis["cost_recommendations"]):
                    markdown_output.append(f"{i+1}. {rec}")
        else:
            markdown_output.append("No cost analysis available.")
        
        markdown_output.append("")
        
        # Add alternatives
        alternatives = recommendation.get("alternatives", [])
        if alternatives:
            markdown_output.append("## Alternative Architectures")
            
            for i, alt in enumerate(alternatives):
                markdown_output.append(f"### Alternative {i+1}: {alt.get('name', '')}")
                markdown_output.append(f"**Provider:** {alt.get('provider', '').upper()}")
                markdown_output.append(f"**Pattern:** {alt.get('pattern', '').replace('_', ' ').title()}")
                markdown_output.append(f"**Suitability Score:** {alt.get('suitability_score', 0):.2f}")
                
                if "components" in alt and alt["components"]:
                    markdown_output.append("")
                    markdown_output.append("#### Components:")
                    
                    alt_component_data = []
                    alt_headers = ["Component", "Provider", "Service", "Tier"]
                    
                    for comp_name, comp_details in alt["components"].items():
                        alt_component_data.append([
                            comp_name.replace("_", " ").title(),
                            comp_details.get("provider", "").upper(),
                            comp_details.get("service_type", ""),
                            comp_details.get("tier", "")
                        ])
                    
                    markdown_output.append(tabulate(alt_component_data, headers=alt_headers, tablefmt="pipe"))
                
                markdown_output.append("")
        
        return "\n".join(markdown_output)
    
    def format_report_as_markdown(self, report: Dict[str, Any]) -> str:
        """
        Format the detailed report as markdown for better readability.
        
        Args:
            report: The report dictionary from the master agent.
            
        Returns:
            Formatted markdown string.
        """
        if not report:
            return "# No report available."
        
        markdown_output = []
        
        # Add title and timestamp
        markdown_output.append("# Cloud Architecture Recommendation Report")
        markdown_output.append(f"*Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        markdown_output.append("")
        
        # Add executive summary
        markdown_output.append("## Executive Summary")
        markdown_output.append(report.get("summary", "No summary available."))
        markdown_output.append("")
        
        # Add detailed recommendation
        if "detailed_recommendation" in report:
            recommendation = report["detailed_recommendation"]
            markdown_output.append("## Selected Architecture")
            
            summary = recommendation.get("recommendation_summary", {})
            summary_table = [
                ["Architecture", summary.get("architecture_name", "N/A")],
                ["Provider", summary.get("provider", "N/A").upper()],
                ["Monthly Cost", f"${summary.get('monthly_cost', 0):.2f}"],
                ["Yearly Cost", f"${summary.get('yearly_cost', 0):.2f}"],
                ["Budget Met", "✅ Yes" if summary.get("budget_met", False) else "❌ No"],
                ["Suitability Score", f"{summary.get('suitability_score', 0):.2f}/1.00"]
            ]
            
            markdown_output.append(tabulate(summary_table, tablefmt="pipe"))
            markdown_output.append("")
        
        # Add architecture diagrams
        if "architecture_diagrams" in report:
            markdown_output.append("## Architecture Diagrams")
            
            for name, diagram in report["architecture_diagrams"].items():
                markdown_output.append(f"### {name.replace('_', ' ').title()} Diagram")
                markdown_output.append(diagram)
                markdown_output.append("")
        
        # Add cost analysis
        if "cost_analysis" in report:
            cost = report["cost_analysis"]
            markdown_output.append("## Cost Analysis")
            
            if "summary" in cost:
                cost_summary_table = [
                    ["Monthly Total", f"${cost['summary'].get('monthly_total', 0):.2f}"],
                    ["Yearly Total", f"${cost['summary'].get('yearly_total', 0):.2f}"],
                    ["3-Year Total", f"${cost['summary'].get('three_year_total', 0):.2f}"],
                    ["5-Year Total", f"${cost['summary'].get('five_year_total', 0):.2f}"]
                ]
                
                markdown_output.append(tabulate(cost_summary_table, tablefmt="pipe"))
                markdown_output.append("")
            
            # Add component costs if available
            if "components" in cost and cost["components"]:
                markdown_output.append("### Cost Components")
                
                component_headers = ["Component", "Monthly Cost", "Yearly Cost", "% of Total"]
                component_data = []
                
                for component in cost["components"]:
                    component_data.append([
                        component.get("name", ""),
                        f"${component.get('monthly_cost', 0):.2f}",
                        f"${component.get('yearly_cost', 0):.2f}",
                        f"{component.get('percentage', 0):.1f}%"
                    ])
                
                markdown_output.append(tabulate(component_data, headers=component_headers, tablefmt="pipe"))
                markdown_output.append("")
        
        # Add implementation steps
        if "implementation_steps" in report:
            markdown_output.append("## Implementation Steps")
            
            for i, step in enumerate(report["implementation_steps"]):
                markdown_output.append(f"### Step {i+1}: {step.get('title', '')}")
                markdown_output.append(step.get("description", ""))
                
                if "sub_steps" in step and step["sub_steps"]:
                    markdown_output.append("")
                    markdown_output.append("**Detailed Steps:**")
                    
                    for j, sub_step in enumerate(step["sub_steps"]):
                        markdown_output.append(f"{j+1}. {sub_step}")
                
                markdown_output.append("")
        
        # Add appendices
        if "appendices" in report and report["appendices"]:
            markdown_output.append("## Appendices")
            
            # Alternative architectures
            if "alternative_architectures" in report["appendices"]:
                markdown_output.append("### Alternative Architectures")
                
                for i, alt in enumerate(report["appendices"]["alternative_architectures"]):
                    if alt.get("name") == "No alternatives available.":
                        markdown_output.append("No alternative architectures available.")
                        continue
                        
                    markdown_output.append(f"#### Alternative {i+1}: {alt.get('name', '')}")
                    markdown_output.append(f"**Provider:** {alt.get('provider', '').upper()}")
                    markdown_output.append(f"**Pattern:** {alt.get('pattern', '').replace('_', ' ').title()}")
                    markdown_output.append(f"**Suitability Score:** {alt.get('suitability_score', 0):.2f}")
                    
                    if "components" in alt and alt["components"]:
                        markdown_output.append("")
                        alt_component_data = []
                        alt_headers = ["Component", "Provider", "Service", "Tier"]
                        
                        for comp_name, comp_details in alt["components"].items():
                            alt_component_data.append([
                                comp_name.replace("_", " ").title(),
                                comp_details.get("provider", "").upper(),
                                comp_details.get("service_type", ""),
                                comp_details.get("tier", "")
                            ])
                        
                        markdown_output.append(tabulate(alt_component_data, headers=alt_headers, tablefmt="pipe"))
                    
                    markdown_output.append("")
        
        return "\n".join(markdown_output)

    def save_markdown_to_file(self, markdown_content: str, file_path: str) -> bool:
        """
        Save markdown content to a file.
        
        Args:
            markdown_content: The markdown content to save.
            file_path: The path where the file should be saved.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"Markdown saved to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving markdown to {file_path}: {str(e)}")
            return False


async def main():
    """
    Main function to run the application.
    """
    parser = argparse.ArgumentParser(description="Cloud Architecture Recommendation System")
    parser.add_argument("--requirements", type=str, help="Path to requirements JSON file")
    parser.add_argument("--output", type=str, help="Path to save the output markdown file")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    
    args = parser.parse_args()
    
    app = CloudArchitectApp()
    await app.initialize()
    
    # Load requirements from file if provided
    if args.requirements:
        try:
            with open(args.requirements, 'r') as f:
                requirements = json.load(f)
        except Exception as e:
            print(f"Error loading requirements file: {str(e)}")
            return
        
        # Run recommendation
        result = await app.run_recommendation(requirements)
        
        if not result:
            print("Failed to generate recommendation.")
            return
        
        # Generate report if requested
        if args.report:
            report_result = await app.generate_report()
            
            if not report_result:
                print("Failed to generate report.")
                return
            
            # Format report as markdown
            markdown_output = app.format_report_as_markdown(report_result)
        else:
            # Format recommendation as markdown
            markdown_output = app.format_recommendation_as_markdown(result.get("recommendation", {}))
        
        # Print or save output
        if args.output:
            app.save_markdown_to_file(markdown_output, args.output)
        else:
            print("\n" + markdown_output)
    else:
        # Interactive mode not implemented here
        # In a real implementation, this would call gather_user_requirements_interactive
        print("Please provide a requirements file using --requirements")
        print("Example: python main.py --requirements requirements.json --output recommendation.md")


if __name__ == "__main__":
    asyncio.run(main())