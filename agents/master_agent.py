"""
Master Agent module that coordinates the multi-agent system and interacts with users.
"""
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, validator

from .base_agent import BaseAgent
from .scraping_agent import ScrapingAgent
from .solution_architect_agent import SolutionArchitectAgent
from .mathematics_agent import MathematicsAgent


class FileType(BaseModel):
    """Model for file type information."""
    monthly_volume: int = Field(..., description="Expected number of files per month")
    avg_size_mb: float = Field(..., description="Average size of each file in MB")


class UserRequirements(BaseModel):
    """Model for user requirements."""
    expected_users: int = Field(..., description="Expected number of users per month")
    max_monthly_budget: float = Field(..., description="Maximum monthly budget in USD")
    file_types: Dict[str, FileType] = Field(
        ..., 
        description="Types of files to be stored with volume and size information"
    )
    preferred_providers: List[str] = Field(
        default=["aws", "gcp", "azure"], 
        description="Preferred cloud providers in order of preference"
    )
    coupling_preference: str = Field(
        default="loose", 
        description="Preference for tight coupling (single provider) or loose coupling (multi-provider)"
    )
    access_pattern: str = Field(
        default="balanced", 
        description="Data access pattern: read_heavy, write_heavy, balanced, analytics"
    )
    transaction_requirements: str = Field(
        default="low", 
        description="Transaction requirements: low, medium, high, critical"
    )
    compliance: List[str] = Field(
        default=[], 
        description="Compliance requirements: hipaa, pci, gdpr, etc."
    )
    data_residency: str = Field(
        default="none", 
        description="Data residency requirements: none, us, eu, etc."
    )
    monthly_user_growth_rate: float = Field(
        default=0.05, 
        description="Expected monthly user growth rate (0.05 = 5%)"
    )
    monthly_storage_growth_rate: float = Field(
        default=0.1, 
        description="Expected monthly storage growth rate (0.1 = 10%)"
    )
    time_horizon_months: int = Field(
        default=12, 
        description="Time horizon for planning in months"
    )
    serverless_preference: bool = Field(
        default=False, 
        description="Preference for serverless architecture"
    )
    requirements: List[str] = Field(
        default=[], 
        description="Additional requirements: strong_consistency, analytics, etc."
    )

    @validator('coupling_preference')
    def validate_coupling_preference(cls, v):
        """Validate coupling preference value."""
        if v not in ["tight", "loose"]:
            raise ValueError("coupling_preference must be 'tight' or 'loose'")
        return v

    @validator('access_pattern')
    def validate_access_pattern(cls, v):
        """Validate access pattern value."""
        valid_patterns = ["read_heavy", "write_heavy", "balanced", "analytics"]
        if v not in valid_patterns:
            raise ValueError(f"access_pattern must be one of {valid_patterns}")
        return v

    @validator('transaction_requirements')
    def validate_transaction_requirements(cls, v):
        """Validate transaction requirements value."""
        valid_requirements = ["low", "medium", "high", "critical"]
        if v not in valid_requirements:
            raise ValueError(f"transaction_requirements must be one of {valid_requirements}")
        return v


class MasterAgent(BaseAgent):
    """
    Agent responsible for orchestrating the multi-agent system and interacting with users.
    
    This agent gathers user requirements, coordinates the activities of the other agents,
    and presents the final recommendations to the user.
    """
    
    def __init__(self, 
                scraping_agent: Optional[ScrapingAgent] = None,
                solution_architect_agent: Optional[SolutionArchitectAgent] = None,
                mathematics_agent: Optional[MathematicsAgent] = None):
        """
        Initialize the master agent with its subagents.
        
        Args:
            scraping_agent: Agent for scraping cloud provider data.
            solution_architect_agent: Agent for generating architecture recommendations.
            mathematics_agent: Agent for validating cost constraints.
        """
        super().__init__(name="MasterAgent")
        
        # Initialize subagents if not provided
        self.scraping_agent = scraping_agent or ScrapingAgent()
        self.solution_architect_agent = solution_architect_agent or SolutionArchitectAgent()
        self.mathematics_agent = mathematics_agent or MathematicsAgent()
        
        # Store the most recent results from each agent
        self.latest_results = {
            "user_requirements": None,
            "pricing_data": None,
            "proposed_architecture": None,
            "cost_analysis": None,
            "final_recommendation": None
        }
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the multi-agent system based on user input.
        
        Args:
            input_data: User requirements and command information.
            
        Returns:
            Results of the multi-agent system's processing.
        """
        # Check if this is a user requirements input or a command
        if "command" in input_data and input_data["command"] == "generate_report":
            return await self._generate_report()
        
        # Process user requirements
        try:
            # Validate and parse user requirements
            user_requirements = self._parse_user_requirements(input_data)
            self.latest_results["user_requirements"] = user_requirements
            
            # Run the multi-agent workflow
            return await self._run_workflow(user_requirements)
            
        except Exception as e:
            self.log_error(f"Error in master agent: {str(e)}")
            return {
                "error": str(e),
                "status": "failed",
                "message": "An error occurred while processing your request."
            }
    
    def _parse_user_requirements(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate user requirements.
        
        Args:
            input_data: Raw user input data.
            
        Returns:
            Validated user requirements dictionary.
        """
        try:
            # Validate with Pydantic model
            user_requirements = UserRequirements(**input_data)
            return user_requirements.dict()
        except Exception as e:
            self.log_error(f"Error parsing user requirements: {str(e)}")
            raise ValueError(f"Invalid user requirements: {str(e)}")
    
    async def _run_workflow(self, user_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete multi-agent workflow.
        
        Args:
            user_requirements: Validated user requirements.
            
        Returns:
            Results of the workflow execution.
        """
        self.log_info("Starting multi-agent workflow")
        
        # 1. Get pricing data from scraping agent
        self.log_info("Requesting pricing data from scraping agent")
        pricing_data = await self.scraping_agent.run(user_requirements)
        
        if "error" in pricing_data:
            self.log_error(f"Error from scraping agent: {pricing_data['error']}")
            return {
                "error": pricing_data["error"],
                "status": "failed",
                "stage": "scraping"
            }
        
        self.latest_results["pricing_data"] = pricing_data
        self.log_info("Received pricing data from scraping agent")
        
        # 2. Get architecture recommendations from solution architect agent
        self.log_info("Requesting architecture recommendations from solution architect agent")
        architect_input = {
            "user_requirements": user_requirements,
            "pricing_data": pricing_data
        }
        
        architecture_recommendations = await self.solution_architect_agent.run(architect_input)
        
        if "error" in architecture_recommendations:
            self.log_error(f"Error from solution architect agent: {architecture_recommendations['error']}")
            return {
                "error": architecture_recommendations["error"],
                "status": "failed",
                "stage": "architecture"
            }
        
        self.latest_results["proposed_architecture"] = architecture_recommendations
        self.log_info("Received architecture recommendations from solution architect agent")
        
        # 3. Validate cost constraints with mathematics agent
        self.log_info("Requesting cost validation from mathematics agent")
        math_input = {
            "user_requirements": user_requirements,
            "proposed_architecture": architecture_recommendations["proposed_architecture"],
            "pricing_data": pricing_data
        }
        
        cost_analysis = await self.mathematics_agent.run(math_input)
        
        if "error" in cost_analysis:
            self.log_error(f"Error from mathematics agent: {cost_analysis['error']}")
            return {
                "error": cost_analysis["error"],
                "status": "failed",
                "stage": "cost_validation"
            }
        
        self.latest_results["cost_analysis"] = cost_analysis
        self.log_info("Received cost validation from mathematics agent")
        
        # 4. Prepare final recommendation
        final_recommendation = self._prepare_final_recommendation(
            user_requirements,
            architecture_recommendations,
            cost_analysis
        )
        
        self.latest_results["final_recommendation"] = final_recommendation
        
        return {
            "status": "success",
            "recommendation": final_recommendation
        }
    
    def _prepare_final_recommendation(self,
                                    user_requirements: Dict[str, Any],
                                    architecture_recommendations: Dict[str, Any],
                                    cost_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the final recommendation by combining results from all agents.
        
        Args:
            user_requirements: User requirements dictionary.
            architecture_recommendations: Results from solution architect agent.
            cost_analysis: Results from mathematics agent.
            
        Returns:
            Final recommendation dictionary.
        """
        # Extract key information
        proposed_architecture = architecture_recommendations["proposed_architecture"]
        explanation = architecture_recommendations.get("explanation", {}).get("best", "")
        
        budget_validation = cost_analysis.get("budget_validation", {})
        tco_results = cost_analysis.get("tco_results", {})
        cost_recommendations = cost_analysis.get("recommendations", [])
        
        # Check if recommendation meets budget
        budget_met = budget_validation.get("budget_met", False)
        
        # Prepare alternatives based on whether budget is met
        alternatives = []
        if not budget_met:
            # Offer more cost-effective alternatives
            alternatives = architecture_recommendations.get("alternative_architectures", [])
        else:
            # Offer alternatives with different trade-offs
            alternatives = architecture_recommendations.get("alternative_architectures", [])
        
        # Build final recommendation
        recommendation = {
            "recommendation_summary": {
                "architecture_name": proposed_architecture.get("name", ""),
                "provider": proposed_architecture.get("provider", ""),
                "monthly_cost": tco_results.get("monthly_total", 0),
                "yearly_cost": tco_results.get("yearly_total", 0),
                "budget_met": budget_met,
                "suitability_score": proposed_architecture.get("suitability_score", 0),
            },
            "proposed_architecture": proposed_architecture,
            "cost_analysis": {
                "monthly_total": tco_results.get("monthly_total", 0),
                "yearly_total": tco_results.get("yearly_total", 0),
                "cost_components": tco_results.get("cost_components", []),
                "budget_validation": budget_validation,
                "cost_recommendations": cost_recommendations
            },
            "explanation": explanation,
            "alternatives": alternatives,
            "user_requirements": user_requirements,
        }
        
        return recommendation
    
    async def _generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report based on the latest results.
        
        Returns:
            Report data dictionary.
        """
        # Check if we have all necessary data
        if not all(value for key, value in self.latest_results.items() if key != "final_recommendation"):
            return {
                "error": "Missing required data. Please run a complete workflow first.",
                "status": "failed",
                "stage": "report_generation"
            }
        
        # Generate report if we don't have a final recommendation yet
        if not self.latest_results["final_recommendation"]:
            self.latest_results["final_recommendation"] = self._prepare_final_recommendation(
                self.latest_results["user_requirements"],
                self.latest_results["proposed_architecture"],
                self.latest_results["cost_analysis"]
            )
        
        # Build report data
        report = {
            "title": "Cloud Architecture Recommendation Report",
            "summary": self._generate_executive_summary(),
            "detailed_recommendation": self.latest_results["final_recommendation"],
            "architecture_diagrams": self._generate_architecture_diagrams(),
            "cost_analysis": self._generate_cost_analysis_report(),
            "implementation_steps": self._generate_implementation_steps(),
            "appendices": {
                "pricing_data_summary": self._generate_pricing_data_summary(),
                "alternative_architectures": self._generate_alternatives_summary()
            }
        }
        
        return {
            "status": "success",
            "report": report
        }
    
    def _generate_executive_summary(self) -> str:
        """
        Generate an executive summary of the recommendation.
        
        Returns:
            Executive summary string.
        """
        if not self.latest_results["final_recommendation"]:
            return "No recommendation available."
        
        final_rec = self.latest_results["final_recommendation"]
        
        # Extract key information
        architecture_name = final_rec["recommendation_summary"]["architecture_name"]
        provider = final_rec["recommendation_summary"]["provider"]
        monthly_cost = final_rec["recommendation_summary"]["monthly_cost"]
        yearly_cost = final_rec["recommendation_summary"]["yearly_cost"]
        budget_met = final_rec["recommendation_summary"]["budget_met"]
        
        # Get user requirements
        user_req = self.latest_results["user_requirements"]
        expected_users = user_req["expected_users"]
        budget = user_req["max_monthly_budget"]
        
        # Generate summary
        summary = f"""
        # Executive Summary
        
        Based on your requirements for supporting {expected_users} users with a monthly budget of ${budget:.2f}, 
        we recommend the **{architecture_name}** architecture on {provider.upper()}.
        
        This solution will cost approximately **${monthly_cost:.2f} per month** (${yearly_cost:.2f} per year)
        and {'meets your budget constraints' if budget_met else 'exceeds your budget constraints by $' + f'{monthly_cost - budget:.2f} per month'}.
        
        The architecture is designed to handle your specific workload characteristics while balancing cost,
        performance, and scalability requirements. It includes components for storage, compute, and database
        needs tailored to your specific use case.
        
        {'## Budget Recommendations' if not budget_met else ''}
        {self._generate_budget_recommendations() if not budget_met else ''}
        """
        
        return summary
    
    def _generate_budget_recommendations(self) -> str:
        """
        Generate budget recommendations if the proposed architecture exceeds budget.
        
        Returns:
            Budget recommendations string.
        """
        if not self.latest_results["cost_analysis"] or "recommendations" not in self.latest_results["cost_analysis"]:
            return "No budget recommendations available."
        
        recommendations = self.latest_results["cost_analysis"]["recommendations"]
        
        if not recommendations:
            return "No specific budget recommendations available."
        
        return "\n".join([f"- {rec}" for rec in recommendations])
    
    def _generate_architecture_diagrams(self) -> Dict[str, str]:
        """
        Generate architecture diagram descriptions.
        
        In a real system, this would generate actual diagrams using a library like diagrams.
        For now, we'll return textual descriptions.
        
        Returns:
            Dictionary of diagram descriptions.
        """
        diagrams = {}
        
        if not self.latest_results["final_recommendation"]:
            return {"main": "No architecture diagram available."}
        
        proposed_architecture = self.latest_results["final_recommendation"]["proposed_architecture"]
        
        # Main architecture diagram description
        diagrams["main"] = f"""
        # {proposed_architecture.get('name', 'Architecture')} Diagram
        
        This architecture consists of the following components:
        
        {self._format_components_for_diagram(proposed_architecture.get('components', {}))}
        
        The components work together to provide a complete solution for your use case.
        """
        
        # Add data flow diagram description
        diagrams["data_flow"] = f"""
        # Data Flow Diagram
        
        This diagram illustrates how data flows between the components in the {proposed_architecture.get('name', 'architecture')}:
        
        1. User requests are received by the entry point service
        2. Authentication and authorization are performed
        3. Data is processed and stored in the appropriate storage systems
        4. Results are returned to the user
        
        The architecture is designed to handle your {self.latest_results['user_requirements'].get('access_pattern', 'balanced')} 
        access pattern efficiently.
        """
        
        return diagrams
    
    def _format_components_for_diagram(self, components: Dict[str, Any]) -> str:
        """
        Format component information for diagram description.
        
        Args:
            components: Dictionary of architecture components.
            
        Returns:
            Formatted component description string.
        """
        if not components:
            return "No components defined."
        
        component_descriptions = []
        
        for name, details in components.items():
            provider = details.get("provider", "")
            service_type = details.get("service_type", "")
            description = details.get("description", "")
            
            component_descriptions.append(f"- **{name.replace('_', ' ').title()}**: {description} ({provider.upper()} {service_type})")
        
        return "\n".join(component_descriptions)
    
    def _generate_cost_analysis_report(self) -> Dict[str, Any]:
        """
        Generate detailed cost analysis report.
        
        Returns:
            Cost analysis report dictionary.
        """
        if not self.latest_results["cost_analysis"]:
            return {"summary": "No cost analysis available."}
        
        cost_analysis = self.latest_results["cost_analysis"]
        tco_results = cost_analysis.get("tco_results", {})
        
        # Extract cost components
        components = tco_results.get("cost_components", [])
        
        # Calculate component percentages
        total_monthly = tco_results.get("monthly_total", 0)
        if total_monthly > 0:
            for component in components:
                component["percentage"] = (component.get("monthly_cost", 0) / total_monthly) * 100
        
        # Generate growth projections description
        growth_projections = tco_results.get("growth_projections", {})
        
        return {
            "summary": {
                "monthly_total": tco_results.get("monthly_total", 0),
                "yearly_total": tco_results.get("yearly_total", 0),
                "three_year_total": tco_results.get("three_year_total", 0),
                "five_year_total": tco_results.get("five_year_total", 0)
            },
            "components": components,
            "growth_projections": growth_projections,
            "budget_validation": cost_analysis.get("budget_validation", {}),
            "cost_recommendations": cost_analysis.get("recommendations", []),
            "sensitivity_analysis": cost_analysis.get("sensitivity_analysis", {})
        }
    
    def _generate_implementation_steps(self) -> List[Dict[str, Any]]:
        """
        Generate implementation steps for the recommended architecture.
        
        Returns:
            List of implementation steps.
        """
        if not self.latest_results["final_recommendation"]:
            return [{"title": "No implementation steps available."}]
        
        architecture = self.latest_results["final_recommendation"]["proposed_architecture"]
        provider = architecture.get("provider", "")
        
        # Generate implementation steps based on provider
        steps = []
        
        # Account setup step
        steps.append({
            "title": f"Set Up {provider.upper()} Account",
            "description": f"Create or configure your {provider.upper()} account with appropriate IAM roles and permissions.",
            "sub_steps": [
                "Create root account or use existing account",
                "Set up billing alerts and budgets",
                "Configure IAM roles and permissions",
                "Set up multi-factor authentication"
            ]
        })
        
        # Infrastructure as code step
        iac_tool = "CloudFormation" if provider == "aws" else "Deployment Manager" if provider == "gcp" else "ARM Templates"
        if provider == "hybrid":
            iac_tool = "Terraform"
            
        steps.append({
            "title": f"Set Up Infrastructure as Code with {iac_tool}",
            "description": f"Define your infrastructure as code using {iac_tool} to automate deployment and ensure consistency.",
            "sub_steps": [
                f"Install and configure {iac_tool}",
                "Create base infrastructure templates",
                "Define variables and parameters",
                "Set up state management"
            ]
        })
        
        # Deploy components step
        steps.append({
            "title": "Deploy Core Infrastructure Components",
            "description": "Deploy the core components of the architecture in the correct order.",
            "sub_steps": [
                "Deploy networking infrastructure (VPC, subnets, etc.)",
                "Set up security groups and firewall rules",
                "Deploy storage components",
                "Deploy database components",
                "Deploy compute components"
            ]
        })
        
        # Configure monitoring and logging
        monitoring_tool = "CloudWatch" if provider == "aws" else "Cloud Monitoring" if provider == "gcp" else "Azure Monitor"
        if provider == "hybrid":
            monitoring_tool = "third-party monitoring solution"
            
        steps.append({
            "title": "Configure Monitoring and Logging",
            "description": f"Set up {monitoring_tool} to monitor your infrastructure and applications.",
            "sub_steps": [
                "Configure metrics and dashboards",
                "Set up log aggregation",
                "Configure alerts and notifications",
                "Test alerting system"
            ]
        })
        
        # Security configurations
        steps.append({
            "title": "Implement Security Configurations",
            "description": "Configure security settings and compliance controls.",
            "sub_steps": [
                "Enable encryption for data at rest and in transit",
                "Implement least privilege access controls",
                "Set up security scanning and compliance checks",
                "Document security policies and procedures"
            ]
        })
        
        # Testing and validation
        steps.append({
            "title": "Test and Validate the Architecture",
            "description": "Perform comprehensive testing to ensure the architecture meets requirements.",
            "sub_steps": [
                "Perform load testing",
                "Validate security controls",
                "Test disaster recovery procedures",
                "Verify compliance requirements are met"
            ]
        })
        
        # Documentation
        steps.append({
            "title": "Create Documentation",
            "description": "Document the architecture, configurations, and operational procedures.",
            "sub_steps": [
                "Create architecture diagrams",
                "Document configuration settings",
                "Create operational runbooks",
                "Document backup and recovery procedures"
            ]
        })
        
        return steps
    
    def _generate_pricing_data_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the pricing data used for calculations.
        
        Returns:
            Pricing data summary dictionary.
        """
        if not self.latest_results["pricing_data"]:
            return {"summary": "No pricing data available."}
        
        pricing_data = self.latest_results["pricing_data"]
        
        # Count services by provider
        service_counts = {}
        for category in ["storage_pricing", "database_pricing", "compute_pricing"]:
            for service_key in pricing_data.get(category, {}):
                provider = service_key.split("_")[0]
                service_counts[provider] = service_counts.get(provider, 0) + 1
        
        # Prepare summary
        return {
            "summary": {
                "scraped_at": pricing_data.get("scraped_at", "Unknown"),
                "providers": list(service_counts.keys()),
                "service_counts": service_counts
            },
            "services_by_category": {
                "storage": list(pricing_data.get("storage_pricing", {}).keys()),
                "database": list(pricing_data.get("database_pricing", {}).keys()),
                "compute": list(pricing_data.get("compute_pricing", {}).keys())
            }
        }
    
    def _generate_alternatives_summary(self) -> List[Dict[str, Any]]:
        """
        Generate summaries of alternative architectures.
        
        Returns:
            List of alternative architecture summaries.
        """
        if not self.latest_results["final_recommendation"] or "alternatives" not in self.latest_results["final_recommendation"]:
            return [{"name": "No alternatives available."}]
        
        alternatives = self.latest_results["final_recommendation"]["alternatives"]
        cost_analysis = self.latest_results["cost_analysis"]
        
        summaries = []
        for i, alt in enumerate(alternatives):
            # Get corresponding explanation if available
            explanation = ""
            if self.latest_results["proposed_architecture"] and "explanation" in self.latest_results["proposed_architecture"]:
                explanation = self.latest_results["proposed_architecture"]["explanation"].get(f"alternative_{i+1}", "")
            
            summaries.append({
                "name": alt.get("name", f"Alternative {i+1}"),
                "provider": alt.get("provider", ""),
                "pattern": alt.get("pattern", ""),
                "components": alt.get("components", {}),
                "suitability_score": alt.get("suitability_score", 0),
                "explanation": explanation
            })
        
        return summaries
    
    async def gather_user_requirements_interactive(self) -> Dict[str, Any]:
        """
        Interactive method to gather user requirements through a conversation.
        
        This would typically be implemented with a CLI or GUI interface.
        
        Returns:
            Completed user requirements dictionary.
        """
        # This is a placeholder for an interactive requirements gathering process
        # In a real implementation, this would use a CLI or GUI to prompt the user
        
        requirements = {
            "expected_users": 1000,
            "max_monthly_budget": 2000.0,
            "file_types": {
                "pdf": {
                    "monthly_volume": 5000,
                    "avg_size_mb": 2.5
                },
                "images": {
                    "monthly_volume": 10000,
                    "avg_size_mb": 1.0
                }
            },
            "preferred_providers": ["aws", "gcp", "azure"],
            "coupling_preference": "loose",
            "access_pattern": "read_heavy",
            "transaction_requirements": "low",
            "compliance": [],
            "data_residency": "none"
        }
        
        return requirements
    
    def save_report_to_file(self, report: Dict[str, Any], file_path: str) -> bool:
        """
        Save a generated report to a JSON file.
        
        Args:
            report: The report dictionary to save.
            file_path: The path where the report should be saved.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.log_info(f"Report saved to {file_path}")
            return True
            
        except Exception as e:
            self.log_error(f"Error saving report to {file_path}: {str(e)}")
            return False
    
    def load_requirements_from_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load user requirements from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing requirements.
            
        Returns:
            Parsed requirements dictionary, or None if loading failed.
        """
        try:
            if not os.path.exists(file_path):
                self.log_error(f"File not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                requirements = json.load(f)
            
            # Validate loaded requirements
            validated_requirements = self._parse_user_requirements(requirements)
            return validated_requirements
            
        except json.JSONDecodeError:
            self.log_error(f"Invalid JSON in file: {file_path}")
            return None
        except Exception as e:
            self.log_error(f"Error loading requirements from {file_path}: {str(e)}")
            return None