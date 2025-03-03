"""
Mathematics Agent module that handles budget calculations and validation.
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_agent import BaseAgent


@dataclass
class CostComponent:
    """Data class representing a cost component in the total cost calculation."""
    name: str
    monthly_cost: float
    yearly_cost: float
    description: str
    scaling_factor: str  # e.g., "per user", "per GB", "fixed"


class MathematicsAgent(BaseAgent):
    """
    Agent responsible for calculating TCO and validating that solutions meet budget constraints.
    
    This agent analyzes the proposed architecture and pricing data to calculate
    comprehensive costs including storage, compute, database, and egress costs.
    """
    
    def __init__(self):
        """Initialize the mathematics agent."""
        super().__init__(name="MathematicsAgent")
        
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate and validate costs for proposed architecture solutions.
        
        Args:
            input_data: Dictionary containing the user requirements, proposed architecture,
                        and pricing data.
            
        Returns:
            Dictionary containing cost analysis and validation results.
        """
        self.log_info("Starting cost calculation and validation")
        
        # Validate input
        required_keys = ["user_requirements", "proposed_architecture", "pricing_data"]
        if not self.validate_input(input_data, required_keys):
            return {"error": "Missing required input data"}
        
        # Extract data
        user_requirements = input_data["user_requirements"]
        proposed_architecture = input_data["proposed_architecture"]
        pricing_data = input_data["pricing_data"]
        
        # Calculate TCO
        tco_results = self._calculate_tco(
            user_requirements,
            proposed_architecture,
            pricing_data
        )
        
        # Validate against budget
        budget = user_requirements.get("max_monthly_budget", float("inf"))
        budget_validation = self._validate_budget(tco_results, budget)
        
        # Generate cost sensitivity analysis
        sensitivity_analysis = self._perform_sensitivity_analysis(
            tco_results,
            user_requirements
        )
        
        return {
            "tco_results": tco_results,
            "budget_validation": budget_validation,
            "sensitivity_analysis": sensitivity_analysis,
            "recommendations": self._generate_cost_recommendations(tco_results, budget)
        }
    
    def _calculate_tco(self,
                     user_requirements: Dict[str, Any],
                     proposed_architecture: Dict[str, Any],
                     pricing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the Total Cost of Ownership for the proposed architecture.
        
        Args:
            user_requirements: User's requirements including expected usage.
            proposed_architecture: The proposed cloud architecture.
            pricing_data: Pricing information for different cloud services.
            
        Returns:
            Dictionary containing detailed cost breakdown.
        """
        # Initialize cost components
        cost_components: List[CostComponent] = []
        
        # Extract user requirements
        expected_users = user_requirements.get("expected_users", 0)
        expected_storage_gb = self._calculate_expected_storage(user_requirements)
        expected_requests = self._calculate_expected_requests(user_requirements)
        expected_egress_gb = self._calculate_expected_egress(user_requirements)
        time_horizon_months = user_requirements.get("time_horizon_months", 12)
        
        # Process each component in the proposed architecture
        for component_name, component in proposed_architecture.get("components", {}).items():
            provider = component.get("provider")
            service_type = component.get("service_type")
            tier = component.get("tier", "standard")
            
            # Calculate costs for this component based on service type
            if "storage" in service_type:
                storage_costs = self._calculate_storage_costs(
                    provider, service_type, tier, expected_storage_gb, pricing_data
                )
                cost_components.extend(storage_costs)
                
            if "database" in service_type:
                database_costs = self._calculate_database_costs(
                    provider, service_type, tier, expected_storage_gb, expected_requests, pricing_data
                )
                cost_components.extend(database_costs)
                
            if "compute" in service_type:
                compute_costs = self._calculate_compute_costs(
                    provider, service_type, tier, expected_users, pricing_data
                )
                cost_components.extend(compute_costs)
                
            # Calculate egress costs for all components
            if expected_egress_gb > 0:
                egress_costs = self._calculate_egress_costs(
                    provider, service_type, expected_egress_gb, pricing_data
                )
                cost_components.extend(egress_costs)
                
        # Calculate growth projections
        growth_projections = self._calculate_growth_projections(
            cost_components, user_requirements, time_horizon_months
        )
        
        # Calculate total costs
        monthly_total = sum(component.monthly_cost for component in cost_components)
        yearly_total = sum(component.yearly_cost for component in cost_components)
        
        # Prepare and return results
        return {
            "cost_components": [
                {
                    "name": component.name,
                    "monthly_cost": component.monthly_cost,
                    "yearly_cost": component.yearly_cost,
                    "description": component.description,
                    "scaling_factor": component.scaling_factor
                }
                for component in cost_components
            ],
            "monthly_total": monthly_total,
            "yearly_total": yearly_total,
            "three_year_total": yearly_total * 3,
            "five_year_total": yearly_total * 5,
            "growth_projections": growth_projections
        }
    
    def _calculate_expected_storage(self, user_requirements: Dict[str, Any]) -> float:
        """
        Calculate the expected storage needs in GB based on user requirements.
        
        Args:
            user_requirements: User's requirements including file types and volumes.
            
        Returns:
            Expected storage in GB.
        """
        total_storage_gb = 0
        
        # Calculate storage based on file types and their expected sizes
        file_types = user_requirements.get("file_types", {})
        for file_type, details in file_types.items():
            # Get number of files and average size per file
            num_files = details.get("monthly_volume", 0)
            avg_size_mb = details.get("avg_size_mb", 0)
            
            # Convert to GB and add to total
            storage_gb = (num_files * avg_size_mb) / 1024  # MB to GB
            total_storage_gb += storage_gb
        
        # Add buffer for system overhead (20%)
        total_storage_gb *= 1.2
        
        return total_storage_gb
    
    def _calculate_expected_requests(self, user_requirements: Dict[str, Any]) -> int:
        """
        Calculate the expected number of API requests based on user requirements.
        
        Args:
            user_requirements: User's requirements including expected users and access patterns.
            
        Returns:
            Expected number of monthly requests.
        """
        # Basic calculation based on users and access patterns
        expected_users = user_requirements.get("expected_users", 0)
        access_frequency = user_requirements.get("access_frequency", "medium")
        
        # Define multipliers for different access frequencies
        frequency_multipliers = {
            "low": 50,       # 50 requests per user per month
            "medium": 250,   # 250 requests per user per month
            "high": 1000,    # 1000 requests per user per month
            "very_high": 5000  # 5000 requests per user per month
        }
        
        multiplier = frequency_multipliers.get(access_frequency, 250)
        
        # Calculate total monthly requests
        total_requests = expected_users * multiplier
        
        return total_requests
    
    def _calculate_expected_egress(self, user_requirements: Dict[str, Any]) -> float:
        """
        Calculate expected data egress in GB based on user requirements.
        
        Args:
            user_requirements: User's requirements including access patterns.
            
        Returns:
            Expected monthly data egress in GB.
        """
        # Basic egress calculation
        expected_users = user_requirements.get("expected_users", 0)
        access_pattern = user_requirements.get("access_pattern", "read_heavy")
        
        # Egress multipliers based on access patterns (GB per user per month)
        egress_multipliers = {
            "read_heavy": 2.0,     # 2 GB per user per month
            "write_heavy": 0.5,    # 0.5 GB per user per month
            "balanced": 1.0,       # 1 GB per user per month
            "analytics": 5.0       # 5 GB per user per month
        }
        
        multiplier = egress_multipliers.get(access_pattern, 1.0)
        
        # Calculate total monthly egress
        total_egress_gb = expected_users * multiplier
        
        return total_egress_gb
    
    def _calculate_storage_costs(self,
                               provider: str,
                               service_type: str,
                               tier: str,
                               storage_gb: float,
                               pricing_data: Dict[str, Any]) -> List[CostComponent]:
        """
        Calculate storage costs for a given cloud provider and service.
        
        Args:
            provider: Cloud provider (aws, gcp, azure).
            service_type: Type of storage service.
            tier: Service tier (standard, premium, etc.).
            storage_gb: Amount of storage needed in GB.
            pricing_data: Pricing information from the scraping agent.
            
        Returns:
            List of cost components related to storage.
        """
        cost_components = []
        
        # Get pricing information for this provider and service
        provider_key = f"{provider}_{service_type}"
        storage_pricing = pricing_data.get("storage_pricing", {}).get(provider_key, {})
        
        if not storage_pricing:
            # Use default pricing if specific pricing not available
            default_rates = {
                "aws": 0.023,      # $0.023 per GB per month for S3 standard
                "gcp": 0.020,      # $0.020 per GB per month for GCS standard
                "azure": 0.018     # $0.018 per GB per month for Azure Blob Storage
            }
            rate = default_rates.get(provider, 0.023)
            monthly_cost = storage_gb * rate
            yearly_cost = monthly_cost * 12
            
            cost_components.append(CostComponent(
                name=f"{provider} {service_type} storage ({tier})",
                monthly_cost=monthly_cost,
                yearly_cost=yearly_cost,
                description=f"{storage_gb:.2f} GB at ${rate:.3f}/GB/month",
                scaling_factor="per GB"
            ))
        else:
            # Use tiered pricing if available
            tier_info = storage_pricing.get("storage_costs", {}).get(tier, {})
            if tier_info:
                # Calculate using tiered rates
                remaining_storage = storage_gb
                total_cost = 0
                
                # Sort tiers by threshold
                tiers = sorted(tier_info.items(), key=lambda x: self._extract_threshold(x[0]))
                
                for tier_name, rate in tiers:
                    threshold = self._extract_threshold(tier_name)
                    if threshold:
                        tier_storage = min(remaining_storage, threshold)
                        tier_cost = tier_storage * rate
                        total_cost += tier_cost
                        remaining_storage -= tier_storage
                        
                        if remaining_storage <= 0:
                            break
                
                monthly_cost = total_cost
                yearly_cost = monthly_cost * 12
                
                cost_components.append(CostComponent(
                    name=f"{provider} {service_type} tiered storage",
                    monthly_cost=monthly_cost,
                    yearly_cost=yearly_cost,
                    description=f"{storage_gb:.2f} GB using tiered pricing",
                    scaling_factor="per GB"
                ))
            else:
                # Fallback to flat rate
                rate = 0.023  # Default rate
                monthly_cost = storage_gb * rate
                yearly_cost = monthly_cost * 12
                
                cost_components.append(CostComponent(
                    name=f"{provider} {service_type} storage (default)",
                    monthly_cost=monthly_cost,
                    yearly_cost=yearly_cost,
                    description=f"{storage_gb:.2f} GB at ${rate:.3f}/GB/month",
                    scaling_factor="per GB"
                ))
        
        return cost_components
    
    def _extract_threshold(self, tier_name: str) -> Optional[float]:
        """
        Extract the threshold value from a tier name like "first_50_tb" or "over_500_tb".
        
        Args:
            tier_name: Name of the pricing tier.
            
        Returns:
            Threshold in GB, or None if not parsable.
        """
        try:
            import re
            match = re.search(r'(\d+)_([a-z]+)', tier_name)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                
                # Convert to GB
                if unit == "gb":
                    return value
                elif unit == "tb":
                    return value * 1024  # TB to GB
                elif unit == "pb":
                    return value * 1024 * 1024  # PB to GB
            
            # Handle "over_X" patterns
            match = re.search(r'over_(\d+)_([a-z]+)', tier_name)
            if match:
                return float("inf")  # No upper limit
                
            return None
        except Exception:
            return None
    
    def _calculate_database_costs(self,
                                provider: str,
                                service_type: str,
                                tier: str,
                                storage_gb: float,
                                monthly_requests: int,
                                pricing_data: Dict[str, Any]) -> List[CostComponent]:
        """
        Calculate database costs for a given cloud provider and service.
        
        Args:
            provider: Cloud provider (aws, gcp, azure).
            service_type: Type of database service.
            tier: Service tier (standard, premium, etc.).
            storage_gb: Amount of storage needed in GB.
            monthly_requests: Number of requests per month.
            pricing_data: Pricing information from the scraping agent.
            
        Returns:
            List of cost components related to database services.
        """
        cost_components = []
        
        # Get pricing information for this provider and service
        provider_key = f"{provider}_{service_type}"
        db_pricing = pricing_data.get("database_pricing", {}).get(provider_key, {})
        
        if not db_pricing:
            # Use default pricing if specific pricing not available
            # Different defaults for relational vs NoSQL databases
            if any(rel_db in service_type for rel_db in ["rds", "sql", "postgres", "mysql"]):
                # Default pricing for relational DBs
                instance_cost = {
                    "aws": 0.095,      # $0.095 per hour for RDS
                    "gcp": 0.0865,     # $0.0865 per hour for Cloud SQL
                    "azure": 0.075     # $0.075 per hour for Azure SQL
                }.get(provider, 0.095)
                
                storage_cost_per_gb = {
                    "aws": 0.115,      # $0.115 per GB per month for RDS
                    "gcp": 0.17,       # $0.17 per GB per month for Cloud SQL
                    "azure": 0.125     # $0.125 per GB per month for Azure SQL
                }.get(provider, 0.115)
                
                # Calculate monthly costs
                monthly_instance_cost = instance_cost * 24 * 30  # cost per hour * 24h * 30 days
                monthly_storage_cost = storage_gb * storage_cost_per_gb
                
                cost_components.append(CostComponent(
                    name=f"{provider} {service_type} instance",
                    monthly_cost=monthly_instance_cost,
                    yearly_cost=monthly_instance_cost * 12,
                    description=f"Database instance at ${instance_cost:.4f}/hour",
                    scaling_factor="fixed"
                ))
                
                cost_components.append(CostComponent(
                    name=f"{provider} {service_type} storage",
                    monthly_cost=monthly_storage_cost,
                    yearly_cost=monthly_storage_cost * 12,
                    description=f"{storage_gb:.2f} GB at ${storage_cost_per_gb:.3f}/GB/month",
                    scaling_factor="per GB"
                ))
            else:
                # Default pricing for NoSQL DBs
                request_cost_per_million = {
                    "aws": 1.25,       # $1.25 per million requests for DynamoDB
                    "gcp": 0.36,       # $0.36 per million requests for Firestore
                    "azure": 0.25      # $0.25 per million requests for Cosmos DB
                }.get(provider, 1.25)
                
                storage_cost_per_gb = {
                    "aws": 0.25,       # $0.25 per GB per month for DynamoDB
                    "gcp": 0.18,       # $0.18 per GB per month for Firestore
                    "azure": 0.25      # $0.25 per GB per month for Cosmos DB
                }.get(provider, 0.25)
                
                # Calculate monthly costs
                monthly_request_cost = (monthly_requests / 1_000_000) * request_cost_per_million
                monthly_storage_cost = storage_gb * storage_cost_per_gb
                
                cost_components.append(CostComponent(
                    name=f"{provider} {service_type} requests",
                    monthly_cost=monthly_request_cost,
                    yearly_cost=monthly_request_cost * 12,
                    description=f"{monthly_requests:,} requests at ${request_cost_per_million:.2f}/million",
                    scaling_factor="per request"
                ))
                
                cost_components.append(CostComponent(
                    name=f"{provider} {service_type} storage",
                    monthly_cost=monthly_storage_cost,
                    yearly_cost=monthly_storage_cost * 12,
                    description=f"{storage_gb:.2f} GB at ${storage_cost_per_gb:.3f}/GB/month",
                    scaling_factor="per GB"
                ))
        else:
            # Use pricing information from the database
            pricing_tiers = db_pricing.get("pricing_tiers", [])
            
            if pricing_tiers:
                # Find the appropriate tier based on storage needs
                selected_tier = None
                for t in pricing_tiers:
                    if t["name"].lower() == tier.lower():
                        selected_tier = t
                        break
                
                # If specified tier not found, select based on storage requirements
                if not selected_tier:
                    for t in pricing_tiers:
                        tier_storage = t.get("storage", "0")
                        if isinstance(tier_storage, str):
                            # Extract the numeric part from strings like "5GB"
                            import re
                            storage_match = re.search(r'(\d+)', tier_storage)
                            if storage_match:
                                tier_storage_gb = float(storage_match.group(1))
                                if "tb" in tier_storage.lower():
                                    tier_storage_gb *= 1024  # TB to GB
                                
                                if tier_storage_gb >= storage_gb:
                                    selected_tier = t
                                    break
                
                # If still no tier found, use the largest one
                if not selected_tier and pricing_tiers:
                    selected_tier = pricing_tiers[-1]
                
                if selected_tier:
                    tier_price = float(selected_tier.get("price", 0))
                    tier_name = selected_tier.get("name", "Unknown")
                    
                    cost_components.append(CostComponent(
                        name=f"{provider} {service_type} ({tier_name} tier)",
                        monthly_cost=tier_price,
                        yearly_cost=tier_price * 12,
                        description=f"{tier_name} tier with {selected_tier.get('storage', 'unknown')} storage",
                        scaling_factor="fixed"
                    ))
            else:
                # Fallback to custom calculation
                monthly_cost = storage_gb * 0.2  # Simple default rate
                
                cost_components.append(CostComponent(
                    name=f"{provider} {service_type} (estimated)",
                    monthly_cost=monthly_cost,
                    yearly_cost=monthly_cost * 12,
                    description=f"Estimated cost based on {storage_gb:.2f} GB storage",
                    scaling_factor="estimated"
                ))
        
        return cost_components
    
    def _calculate_compute_costs(self,
                               provider: str,
                               service_type: str,
                               tier: str,
                               expected_users: int,
                               pricing_data: Dict[str, Any]) -> List[CostComponent]:
        """
        Calculate compute costs for a given cloud provider and service.
        
        Args:
            provider: Cloud provider (aws, gcp, azure).
            service_type: Type of compute service.
            tier: Service tier (standard, premium, etc.).
            expected_users: Number of expected users.
            pricing_data: Pricing information from the scraping agent.
            
        Returns:
            List of cost components related to compute services.
        """
        cost_components = []
        
        # Compute requirements scale with user count
        # Simplified sizing logic:
        # - 1 vCPU per 500 users
        # - 4 GB RAM per vCPU
        
        vcpus_needed = max(1, math.ceil(expected_users / 500))
        ram_gb_needed = vcpus_needed * 4
        
        # Default prices per hour for compute instances
        compute_hourly_rates = {
            "aws": 0.0416,      # $0.0416 per hour for t3.medium (2 vCPU, 4 GB RAM)
            "gcp": 0.0350,      # $0.0350 per hour for e2-standard-2 (2 vCPU, 8 GB RAM)
            "azure": 0.0456     # $0.0456 per hour for B2s (2 vCPU, 4 GB RAM)
        }
        
        # Adjust for number of vCPUs needed (simplified)
        hourly_rate = compute_hourly_rates.get(provider, 0.04) * (vcpus_needed / 2)
        monthly_cost = hourly_rate * 24 * 30  # cost per hour * 24h * 30 days
        
        cost_components.append(CostComponent(
            name=f"{provider} {service_type} compute",
            monthly_cost=monthly_cost,
            yearly_cost=monthly_cost * 12,
            description=f"{vcpus_needed} vCPUs, {ram_gb_needed} GB RAM at ${hourly_rate:.4f}/hour",
            scaling_factor="per user"
        ))
        
        return cost_components
    
    def _calculate_egress_costs(self,
                              provider: str,
                              service_type: str,
                              egress_gb: float,
                              pricing_data: Dict[str, Any]) -> List[CostComponent]:
        """
        Calculate data egress costs for a given cloud provider and service.
        
        Args:
            provider: Cloud provider (aws, gcp, azure).
            service_type: Type of service.
            egress_gb: Amount of egress data in GB.
            pricing_data: Pricing information from the scraping agent.
            
        Returns:
            List of cost components related to data egress.
        """
        cost_components = []
        
        # Default egress rates per GB
        egress_rates = {
            "aws": 0.09,      # $0.09 per GB for AWS
            "gcp": 0.12,      # $0.12 per GB for GCP
            "azure": 0.087    # $0.087 per GB for Azure
        }
        
        # Get provider-specific pricing if available
        provider_key = f"{provider}_{service_type}"
        service_pricing = None
        
        # Check different service categories
        for category in ["storage_pricing", "database_pricing", "compute_pricing"]:
            if provider_key in pricing_data.get(category, {}):
                service_pricing = pricing_data[category][provider_key]
                break
        
        if service_pricing and "egress_costs" in service_pricing:
            # Use the "internet" egress rate if available
            rate = service_pricing["egress_costs"].get("internet", egress_rates.get(provider, 0.09))
        else:
            # Fallback to default rate
            rate = egress_rates.get(provider, 0.09)
        
        monthly_cost = egress_gb * rate
        
        cost_components.append(CostComponent(
            name=f"{provider} data egress",
            monthly_cost=monthly_cost,
            yearly_cost=monthly_cost * 12,
            description=f"{egress_gb:.2f} GB at ${rate:.3f}/GB",
            scaling_factor="per GB"
        ))
        
        return cost_components
    
    def _calculate_growth_projections(self,
                                    cost_components: List[CostComponent],
                                    user_requirements: Dict[str, Any],
                                    time_horizon_months: int) -> Dict[str, List[float]]:
        """
        Calculate cost growth projections over time based on growth rates.
        
        Args:
            cost_components: List of cost components.
            user_requirements: User's requirements including growth rates.
            time_horizon_months: Number of months to project.
            
        Returns:
            Dictionary with monthly and cumulative cost projections.
        """
        # Get monthly growth rates
        user_growth_rate = user_requirements.get("monthly_user_growth_rate", 0.05)  # 5% default
        storage_growth_rate = user_requirements.get("monthly_storage_growth_rate", 0.1)  # 10% default
        
        # Initialize projections
        monthly_costs = []
        cumulative_costs = []
        
        # Calculate cost components with different scaling factors
        fixed_costs = sum(c.monthly_cost for c in cost_components if c.scaling_factor == "fixed")
        per_user_costs = sum(c.monthly_cost for c in cost_components if c.scaling_factor == "per user")
        per_gb_costs = sum(c.monthly_cost for c in cost_components if c.scaling_factor == "per GB")
        per_request_costs = sum(c.monthly_cost for c in cost_components if c.scaling_factor == "per request")
        
        # Base multipliers
        user_multiplier = 1.0
        storage_multiplier = 1.0
        
        # Project costs for each month
        cumulative_cost = 0
        for month in range(time_horizon_months):
            # Calculate costs with growth
            monthly_cost = (
                fixed_costs +
                per_user_costs * user_multiplier +
                per_gb_costs * storage_multiplier +
                per_request_costs * user_multiplier  # Requests scale with users
            )
            
            monthly_costs.append(round(monthly_cost, 2))
            cumulative_cost += monthly_cost
            cumulative_costs.append(round(cumulative_cost, 2))
            
            # Increase multipliers for next month
            user_multiplier *= (1 + user_growth_rate)
            storage_multiplier *= (1 + storage_growth_rate)
        
        return {
            "monthly_costs": monthly_costs,
            "cumulative_costs": cumulative_costs,
            "months": list(range(1, time_horizon_months + 1))
        }
    
    def _validate_budget(self, tco_results: Dict[str, Any], budget: float) -> Dict[str, Any]:
        """
        Validate if the proposed architecture meets the budget constraints.
        
        Args:
            tco_results: The results from TCO calculation.
            budget: The monthly budget constraint.
            
        Returns:
            Validation results with budget compliance information.
        """
        monthly_total = tco_results.get("monthly_total", 0)
        budget_met = monthly_total <= budget
        
        # Calculate how much under/over budget
        if budget > 0:
            budget_percentage = (monthly_total / budget) * 100
            margin = budget - monthly_total
        else:
            budget_percentage = float('inf')
            margin = float('-inf')
        
        return {
            "budget_met": budget_met,
            "monthly_budget": budget,
            "monthly_cost": monthly_total,
            "budget_percentage": round(budget_percentage, 2),
            "margin": round(margin, 2)
        }
    
    def _perform_sensitivity_analysis(self,
                                    tco_results: Dict[str, Any],
                                    user_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on cost factors.
        
        Args:
            tco_results: The results from TCO calculation.
            user_requirements: User's requirements including growth rates.
            
        Returns:
            Dictionary containing sensitivity analysis results.
        """
        # Identify top cost drivers
        cost_components = tco_results.get("cost_components", [])
        if not cost_components:
            return {"error": "No cost components available for analysis"}
        
        # Sort components by monthly cost
        sorted_components = sorted(
            cost_components, 
            key=lambda x: x.get("monthly_cost", 0), 
            reverse=True
        )
        
        # Top cost drivers (top 3 or all if fewer)
        top_drivers = sorted_components[:min(3, len(sorted_components))]
        
        # Calculate what happens if these drivers increase/decrease by 20%
        sensitivity_results = []
        for driver in top_drivers:
            driver_name = driver.get("name", "Unknown")
            driver_cost = driver.get("monthly_cost", 0)
            
            # Calculate impact of 20% increase
            increase_impact = driver_cost * 0.2
            increase_percentage = (increase_impact / tco_results.get("monthly_total", 1)) * 100
            
            sensitivity_results.append({
                "component": driver_name,
                "monthly_cost": driver_cost,
                "increase_impact": round(increase_impact, 2),
                "increase_percentage": round(increase_percentage, 2)
            })
        
        # Calculate breakeven point for number of users
        expected_users = user_requirements.get("expected_users", 0)
        monthly_total = tco_results.get("monthly_total", 0)
        budget = user_requirements.get("max_monthly_budget", float('inf'))
        
        if expected_users > 0 and monthly_total > 0:
            cost_per_user = monthly_total / expected_users
            if budget < float('inf'):
                breakeven_users = math.floor(budget / cost_per_user)
            else:
                breakeven_users = None
        else:
            cost_per_user = None
            breakeven_users = None
        
        return {
            "top_cost_drivers": sensitivity_results,
            "cost_per_user": round(cost_per_user, 2) if cost_per_user else None,
            "breakeven_users": breakeven_users
        }
    
    def _generate_cost_recommendations(self,
                                     tco_results: Dict[str, Any],
                                     budget: float) -> List[str]:
        """
        Generate cost optimization recommendations based on TCO analysis.
        
        Args:
            tco_results: The results from TCO calculation.
            budget: The monthly budget constraint.
            
        Returns:
            List of cost optimization recommendations.
        """
        recommendations = []
        monthly_total = tco_results.get("monthly_total", 0)
        cost_components = tco_results.get("cost_components", [])
        
        # Check if over budget
        if budget < float('inf') and monthly_total > budget:
            over_budget_amount = monthly_total - budget
            recommendations.append(
                f"The proposed architecture is ${over_budget_amount:.2f} over budget. "
                f"Consider the following optimizations."
            )
        
        # Sort components by monthly cost
        sorted_components = sorted(
            cost_components, 
            key=lambda x: x.get("monthly_cost", 0), 
            reverse=True
        )
        
        # Analyze top cost drivers
        if sorted_components:
            top_driver = sorted_components[0]
            top_driver_name = top_driver.get("name", "Unknown")
            top_driver_scaling = top_driver.get("scaling_factor", "unknown")
            
            recommendations.append(
                f"Your highest cost component is {top_driver_name}, which scales with {top_driver_scaling}. "
                f"Consider optimizing this component first."
            )
        
        # Check for specific optimization opportunities
        for component in cost_components:
            name = component.get("name", "").lower()
            scaling = component.get("scaling_factor", "")
            
            # Storage optimization recommendations
            if "storage" in name and scaling == "per GB":
                recommendations.append(
                    "Consider implementing data lifecycle policies to move infrequently accessed "
                    "data to cheaper storage tiers."
                )
                break
        
        # Reserved instances recommendation
        compute_components = [c for c in cost_components if "compute" in c.get("name", "").lower()]
        if compute_components and sum(c.get("monthly_cost", 0) for c in compute_components) > 100:
            recommendations.append(
                "Consider using reserved instances for compute resources to save up to 60% "
                "if you can commit to 1-3 years."
            )
        
        # Regional recommendation
        if monthly_total > 1000:
            recommendations.append(
                "Consider deploying in a different region as prices can vary by 15-30% between regions."
            )
        
        # Add generic recommendations if specific ones didn't apply
        if len(recommendations) < 3:
            recommendations.append(
                "Regularly monitor and analyze your usage patterns to identify opportunities "
                "for rightsizing resources."
            )
            
            recommendations.append(
                "Implement auto-scaling to match resources with demand and avoid "
                "paying for idle capacity."
            )
        
        return recommendations