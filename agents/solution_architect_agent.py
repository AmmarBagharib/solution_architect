"""
Solution Architect Agent module that uses DeepSeek to generate and evaluate cloud architecture designs.
"""
import asyncio
import json
import os
import logging
from typing import Any, Dict, List, Optional, Tuple

# For DeepSeek API integration
import aiohttp
from dotenv import load_dotenv

# Load environment variables including DeepSeek API key
load_dotenv()

# Optional fallback if transformers is available for local model
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import pandas as pd
import numpy as np

from .base_agent import BaseAgent


class SolutionArchitectAgent(BaseAgent):
    """
    Agent responsible for designing and recommending cloud architecture solutions.
    
    This agent uses DeepSeek API to analyze requirements and generate
    appropriate cloud architecture recommendations.
    """
    
    def __init__(self, use_api: bool = True, model_name: str = "deepseek-reasoner"):
        """
        Initialize the solution architect agent with a reasoning model.
        
        Args:
            use_api: Whether to use the DeepSeek API (True) or local model (False)
            model_name: Model name to use with DeepSeek API or local model path
        """
        super().__init__(name="SolutionArchitectAgent")
        
        # DeepSeek API configuration
        self.use_api = use_api
        self.model_name = model_name
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        
        if self.use_api and not self.api_key:
            self.log_error("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")
            self.use_api = False
        
        # Architecture pattern examples to provide context to DeepSeek
        self.architecture_examples = self._load_architecture_examples()
        
        # Initialize local model if needed (will be lazy-loaded when first needed)
        self.model = None
        self.tokenizer = None
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate architecture recommendations based on user requirements.
        
        Args:
            input_data: Contains user requirements and pricing data.
            
        Returns:
            Dictionary containing architecture recommendations.
        """
        self.log_info("Starting architecture design process with DeepSeek")
        
        # Validate input
        required_keys = ["user_requirements", "pricing_data"]
        if not self.validate_input(input_data, required_keys):
            return {"error": "Missing required input data"}
        
        # Extract data
        user_requirements = input_data["user_requirements"]
        pricing_data = input_data["pricing_data"]
        
        try:
            # Generate primary architecture recommendation using DeepSeek
            primary_recommendation = await self._generate_primary_recommendation(
                user_requirements,
                pricing_data
            )
            
            # Generate alternative architecture recommendations
            alternative_recommendations = await self._generate_alternative_recommendations(
                user_requirements,
                pricing_data,
                primary_recommendation
            )
            
            # Generate explanations for each recommendation
            explanation = await self._generate_explanations(
                primary_recommendation,
                alternative_recommendations,
                user_requirements
            )
            
            return {
                "proposed_architecture": primary_recommendation,
                "alternative_architectures": alternative_recommendations,
                "explanation": explanation,
                "workload_characteristics": self._extract_workload_characteristics(user_requirements)
            }
            
        except Exception as e:
            self.log_error(f"Error in architecture design process: {str(e)}")
            return {"error": f"Failed to generate architecture recommendations: {str(e)}"}
    
    def _load_architecture_examples(self) -> Dict[str, Dict[str, Any]]:
        """
        Load example architecture patterns to provide context to DeepSeek.
        
        Returns:
            Dictionary of architecture examples organized by pattern type.
        """
        return {
            "document_storage": {
                "description": "Architecture pattern for document storage systems",
                "components": {
                    "storage": {
                        "description": "Object storage for documents",
                        "aws": "s3",
                        "gcp": "cloud_storage",
                        "azure": "blob_storage"
                    },
                    "metadata_db": {
                        "description": "Database for document metadata",
                        "aws": "dynamodb",
                        "gcp": "firestore",
                        "azure": "cosmos_db"
                    },
                    "search": {
                        "description": "Search service for document contents",
                        "aws": "opensearch",
                        "gcp": "vertex_ai_search",
                        "azure": "cognitive_search"
                    }
                }
            },
            "web_application": {
                "description": "Architecture pattern for web applications",
                "components": {
                    "web_server": {
                        "description": "Web server for application frontend",
                        "aws": "ec2",
                        "gcp": "compute_engine",
                        "azure": "virtual_machines"
                    },
                    "database": {
                        "description": "Database for application data",
                        "aws": "rds",
                        "gcp": "cloud_sql",
                        "azure": "sql_database"
                    },
                    "cache": {
                        "description": "Cache for session data",
                        "aws": "elasticache",
                        "gcp": "memorystore",
                        "azure": "redis_cache"
                    }
                }
            },
            "serverless": {
                "description": "Architecture pattern for serverless applications",
                "components": {
                    "functions": {
                        "description": "Serverless functions for application logic",
                        "aws": "lambda",
                        "gcp": "cloud_functions",
                        "azure": "functions"
                    },
                    "storage": {
                        "description": "Object storage for application data",
                        "aws": "s3",
                        "gcp": "cloud_storage",
                        "azure": "blob_storage"
                    },
                    "database": {
                        "description": "Serverless database for application data",
                        "aws": "dynamodb",
                        "gcp": "firestore",
                        "azure": "cosmos_db"
                    }
                }
            }
        }
    
    def _extract_workload_characteristics(self, user_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract basic workload characteristics from user requirements.
        
        Args:
            user_requirements: Dictionary of user requirements.
            
        Returns:
            Dictionary of workload characteristics.
        """
        return {
            "data_access_pattern": user_requirements.get("access_pattern", "balanced"),
            "transaction_requirements": user_requirements.get("transaction_requirements", "low"),
            "scaling_needs": "high" if user_requirements.get("expected_users", 0) > 10000 else 
                            "medium" if user_requirements.get("expected_users", 0) > 1000 else "low",
            "data_locality_requirements": user_requirements.get("data_residency", "none"),
            "compliance_requirements": user_requirements.get("compliance", [])
        }
    
    async def _generate_primary_recommendation(self, 
                                            user_requirements: Dict[str, Any],
                                            pricing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate primary architecture recommendation using DeepSeek.
        
        Args:
            user_requirements: User requirements dictionary.
            pricing_data: Pricing data dictionary.
            
        Returns:
            Primary architecture recommendation.
        """
        self.log_info("Generating primary architecture recommendation with DeepSeek")
        
        # Create a prompt for DeepSeek to generate the primary recommendation
        prompt = self._create_primary_recommendation_prompt(user_requirements, pricing_data)
        
        # Get response from DeepSeek
        response = await self._get_llm_response(prompt)
        
        # Parse the recommendation from the response
        recommendation = self._parse_architecture_recommendation(response)
        
        if not recommendation:
            self.log_error("Failed to parse primary recommendation from DeepSeek response")
            # Create a fallback recommendation
            recommendation = self._create_fallback_recommendation(user_requirements)
        
        return recommendation
    
    async def _generate_alternative_recommendations(self,
                                                 user_requirements: Dict[str, Any],
                                                 pricing_data: Dict[str, Any],
                                                 primary_recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate alternative architecture recommendations using DeepSeek.
        
        Args:
            user_requirements: User requirements dictionary.
            pricing_data: Pricing data dictionary.
            primary_recommendation: Primary architecture recommendation.
            
        Returns:
            List of alternative architecture recommendations.
        """
        self.log_info("Generating alternative architecture recommendations with DeepSeek")
        
        # Create a prompt for DeepSeek to generate alternative recommendations
        prompt = self._create_alternative_recommendations_prompt(
            user_requirements, 
            pricing_data, 
            primary_recommendation
        )
        
        # Get response from DeepSeek
        response = await self._get_llm_response(prompt)
        
        # Parse the alternative recommendations from the response
        alternatives = self._parse_alternative_recommendations(response)
        
        # Limit to 2 alternatives
        return alternatives[:2]
    
    async def _generate_explanations(self,
                                  primary_recommendation: Dict[str, Any],
                                  alternative_recommendations: List[Dict[str, Any]],
                                  user_requirements: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate detailed explanations for recommendations using DeepSeek.
        
        Args:
            primary_recommendation: Primary architecture recommendation.
            alternative_recommendations: List of alternative recommendations.
            user_requirements: User requirements dictionary.
            
        Returns:
            Dictionary of explanations for each recommendation.
        """
        self.log_info("Generating detailed explanations with DeepSeek")
        
        explanations = {}
        
        # Generate explanation for primary recommendation
        primary_prompt = self._create_explanation_prompt(
            primary_recommendation, 
            user_requirements,
            is_primary=True
        )
        primary_explanation = await self._get_llm_response(primary_prompt)
        explanations["best"] = primary_explanation
        
        # Generate explanations for alternatives
        for i, alt in enumerate(alternative_recommendations):
            alt_prompt = self._create_explanation_prompt(
                alt,
                user_requirements,
                is_primary=False
            )
            alt_explanation = await self._get_llm_response(alt_prompt)
            explanations[f"alternative_{i+1}"] = alt_explanation
        
        return explanations
    
    def _create_primary_recommendation_prompt(self, 
                                           user_requirements: Dict[str, Any],
                                           pricing_data: Dict[str, Any]) -> str:
        """
        Create a prompt for DeepSeek to generate the primary recommendation.
        
        Args:
            user_requirements: User requirements dictionary.
            pricing_data: Pricing data dictionary.
            
        Returns:
            Prompt string.
        """
        # Format requirements and examples for the prompt
        requirements_json = json.dumps(user_requirements, indent=2)
        examples_json = json.dumps(self.architecture_examples, indent=2)
        
        # Create the pricing data summary
        pricing_summary = "Pricing data is available for the following services:\n"
        for category in pricing_data:
            if isinstance(pricing_data[category], dict) and category != "raw_data":
                pricing_summary += f"\n{category.replace('_', ' ').title()}:\n"
                for service in pricing_data[category]:
                    pricing_summary += f"- {service}\n"
        
        prompt = f"""
        You are an expert cloud solution architect. Your task is to recommend the most appropriate cloud architecture 
        for a user based on their requirements. 
        
        The user has the following requirements:
        ```json
        {requirements_json}
        ```
        
        {pricing_summary}
        
        Here are examples of common cloud architecture patterns to help you:
        ```json
        {examples_json}
        ```
        
        Based on the user requirements, design the optimal cloud architecture following these steps:
        
        1. Analyze the requirements to determine the primary use case and workload characteristics
        2. Select the most appropriate architecture pattern and cloud provider(s)
        3. Design a detailed architecture with specific components and services
        4. Assign a suitability score between 0.0 and 1.0 to indicate how well the architecture meets the requirements
        
        Provide your recommendation in JSON format with the following structure:
        ```json
        {{
          "name": "Name of the architecture",
          "provider": "aws, gcp, azure, or hybrid",
          "pattern": "Pattern name (e.g., document_storage, web_application, serverless, etc.)",
          "components": {{
            "component_name": {{
              "description": "Description of the component",
              "provider": "Cloud provider for this component",
              "service_type": "Specific service name",
              "tier": "Service tier/size"
            }},
            ...
          }},
          "description": "Brief description of the architecture",
          "suitability_score": 0.95  // Score between 0.0 and 1.0
        }}
        ```
        
        Your response should only contain the JSON without any other text. Ensure all fields are properly filled.
        """
        
        return prompt
    
    def _create_alternative_recommendations_prompt(self,
                                                user_requirements: Dict[str, Any],
                                                pricing_data: Dict[str, Any],
                                                primary_recommendation: Dict[str, Any]) -> str:
        """
        Create a prompt for DeepSeek to generate alternative recommendations.
        
        Args:
            user_requirements: User requirements dictionary.
            pricing_data: Pricing data dictionary.
            primary_recommendation: Primary architecture recommendation.
            
        Returns:
            Prompt string.
        """
        # Format requirements and primary recommendation for the prompt
        requirements_json = json.dumps(user_requirements, indent=2)
        primary_json = json.dumps(primary_recommendation, indent=2)
        
        prompt = f"""
        You are an expert cloud solution architect. Your primary recommendation for the user's requirements has been made,
        but now we need to generate 2 alternative architecture recommendations that provide different trade-offs
        or leverage different cloud providers.
        
        The user has the following requirements:
        ```json
        {requirements_json}
        ```
        
        The primary recommendation is:
        ```json
        {primary_json}
        ```
        
        Please generate 2 alternative architecture recommendations that:
        1. Use different cloud providers or a different architecture pattern than the primary recommendation
        2. Offer different trade-offs (e.g., cost vs. performance, simplicity vs. flexibility)
        3. Still reasonably satisfy the user's requirements
        
        For each alternative, assign a suitability score between 0.0 and 1.0 that is slightly lower than the primary recommendation.
        
        Provide your recommendations as a JSON array with the following structure:
        ```json
        [
          {{
            "name": "Name of the architecture",
            "provider": "aws, gcp, azure, or hybrid",
            "pattern": "Pattern name (e.g., document_storage, web_application, serverless, etc.)",
            "components": {{
              "component_name": {{
                "description": "Description of the component",
                "provider": "Cloud provider for this component",
                "service_type": "Specific service name",
                "tier": "Service tier/size"
              }},
              ...
            }},
            "description": "Brief description of the architecture",
            "suitability_score": 0.85  // Score between 0.0 and 1.0
          }},
          {{
            // Second alternative recommendation
          }}
        ]
        ```
        
        Your response should only contain the JSON array without any other text. Ensure all fields are properly filled.
        """
        
        return prompt
    
    def _create_explanation_prompt(self,
                                 recommendation: Dict[str, Any],
                                 user_requirements: Dict[str, Any],
                                 is_primary: bool) -> str:
        """
        Create a prompt for DeepSeek to generate an explanation for a recommendation.
        
        Args:
            recommendation: Architecture recommendation to explain.
            user_requirements: User requirements dictionary.
            is_primary: Whether this is the primary recommendation.
            
        Returns:
            Prompt string.
        """
        # Format recommendation and requirements for the prompt
        recommendation_json = json.dumps(recommendation, indent=2)
        requirements_json = json.dumps(user_requirements, indent=2)
        
        architecture_type = "recommended" if is_primary else "alternative"
        
        prompt = f"""
        You are an expert cloud solution architect explaining a {architecture_type} architecture to a user.
        
        The user has the following requirements:
        ```json
        {requirements_json}
        ```
        
        The {architecture_type} architecture is:
        ```json
        {recommendation_json}
        ```
        
        Provide a detailed explanation of this architecture, including:
        
        1. An overview of the architecture and why it was chosen
        2. A detailed description of each component and its purpose
        3. How the architecture aligns with the user's specific requirements
        4. The strengths and limitations of this architecture
        5. Implementation considerations and best practices
        
        Format your explanation in Markdown with appropriate headings and bullet points. Make it comprehensive yet
        easy to understand for someone with basic cloud knowledge.
        
        Begin with a heading "## {"Recommended" if is_primary else "Alternative"} Architecture: [Architecture Name]"
        """
        
        return prompt
    
    def _parse_architecture_recommendation(self, response: str) -> Dict[str, Any]:
        """
        Parse the architecture recommendation from the DeepSeek response.
        
        Args:
            response: Response from DeepSeek.
            
        Returns:
            Parsed architecture recommendation or empty dict if parsing fails.
        """
        try:
            # Find the JSON object in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                self.log_error("No JSON found in DeepSeek response")
                return {}
            
            json_text = response[json_start:json_end]
            recommendation = json.loads(json_text)
            
            # Ensure the recommendation has all required fields
            required_fields = ["name", "provider", "pattern", "components", "description", "suitability_score"]
            for field in required_fields:
                if field not in recommendation:
                    self.log_error(f"Missing required field '{field}' in recommendation")
                    return {}
            
            return recommendation
            
        except Exception as e:
            self.log_error(f"Error parsing architecture recommendation: {str(e)}")
            return {}
    
    def _parse_alternative_recommendations(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the alternative recommendations from the DeepSeek response.
        
        Args:
            response: Response from DeepSeek.
            
        Returns:
            List of parsed alternative recommendations.
        """
        try:
            # Find the JSON array in the response
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            
            if json_start == -1 or json_end == 0:
                # Try to find a single JSON object instead
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                
                if json_start == -1 or json_end == 0:
                    self.log_error("No JSON found in DeepSeek response for alternatives")
                    return []
                
                # Try to parse as a single object
                json_text = response[json_start:json_end]
                alternative = json.loads(json_text)
                return [alternative]
            
            # Parse as an array
            json_text = response[json_start:json_end]
            alternatives = json.loads(json_text)
            
            # Ensure it's a list
            if not isinstance(alternatives, list):
                alternatives = [alternatives]
            
            # Validate each alternative
            valid_alternatives = []
            for alt in alternatives:
                # Check required fields
                required_fields = ["name", "provider", "pattern", "components", "description", "suitability_score"]
                if all(field in alt for field in required_fields):
                    valid_alternatives.append(alt)
                else:
                    self.log_error(f"Skipping alternative with missing fields: {alt.get('name', 'unnamed')}")
            
            return valid_alternatives
            
        except Exception as e:
            self.log_error(f"Error parsing alternative recommendations: {str(e)}")
            return []
    
    def _create_fallback_recommendation(self, user_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a fallback recommendation if DeepSeek fails to generate one.
        
        Args:
            user_requirements: User requirements dictionary.
            
        Returns:
            Fallback architecture recommendation.
        """
        # Determine preferred provider
        preferred_providers = user_requirements.get("preferred_providers", ["aws", "gcp", "azure"])
        provider = preferred_providers[0] if preferred_providers else "aws"
        
        # Determine architecture pattern based on file types
        file_types = user_requirements.get("file_types", {})
        
        if any(ft in ["pdf", "documents", "images"] for ft in file_types):
            pattern = "document_storage"
            components = {
                "storage": {
                    "description": "Object storage for documents",
                    "provider": provider,
                    "service_type": "s3" if provider == "aws" else "cloud_storage" if provider == "gcp" else "blob_storage",
                    "tier": "standard"
                },
                "metadata_db": {
                    "description": "Database for document metadata",
                    "provider": provider,
                    "service_type": "dynamodb" if provider == "aws" else "firestore" if provider == "gcp" else "cosmos_db",
                    "tier": "standard"
                }
            }
        else:
            pattern = "web_application"
            components = {
                "web_server": {
                    "description": "Web server for application frontend",
                    "provider": provider,
                    "service_type": "ec2" if provider == "aws" else "compute_engine" if provider == "gcp" else "virtual_machines",
                    "tier": "standard"
                },
                "database": {
                    "description": "Database for application data",
                    "provider": provider,
                    "service_type": "rds" if provider == "aws" else "cloud_sql" if provider == "gcp" else "sql_database",
                    "tier": "standard"
                }
            }
        
        return {
            "name": f"{provider.upper()} {pattern.replace('_', ' ').title()} Architecture",
            "provider": provider,
            "pattern": pattern,
            "components": components,
            "description": f"A {pattern.replace('_', ' ')} architecture using {provider.upper()} services.",
            "suitability_score": 0.7  # Conservative score for fallback
        }
    
    async def _get_llm_response(self, prompt: str) -> str:
        """
        Get a response from the LLM (DeepSeek API or local model).
        
        Args:
            prompt: The prompt to send to the LLM.
            
        Returns:
            The LLM's response as a string.
        """
        if self.use_api and self.api_key:
            return await self._call_deepseek_api(prompt)
        else:
            return "deepseek API call failed"
    
    async def _call_deepseek_api(self, prompt: str) -> str:
        """
        Call the DeepSeek API to get a response.
        
        Args:
            prompt: The prompt to send to the API.
            
        Returns:
            The API's response text.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,  # Lower temperature for more consistent responses
            "max_tokens": 4000    # Increased token limit for detailed responses
        }
        
        try:
            self.log_info(f"Calling DeepSeek API with model: {self.model_name}")
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.log_info("Received response from DeepSeek API")
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        self.log_error(f"DeepSeek API error: {response.status}, {error_text}")
                        raise Exception(f"API call failed with status {response.status}: {error_text}")
        except aiohttp.ClientError as e:
            self.log_error(f"DeepSeek API request error: {str(e)}")
            raise Exception(f"API connection error: {str(e)}")