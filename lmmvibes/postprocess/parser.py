"""
JSON parsing stage for extracted properties.

This stage migrates the parsing logic from post_processing.py into the pipeline architecture.
"""

import json
import uuid
from typing import Dict, Any, Optional, List
from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset, Property
from ..core.mixins import LoggingMixin, TimingMixin, ErrorHandlingMixin, WandbMixin


class LLMJsonParser(LoggingMixin, TimingMixin, ErrorHandlingMixin, WandbMixin, PipelineStage):
    """
    Parse JSON responses from LLM property extraction.
    
    This stage takes raw LLM responses and parses them into structured Property objects.
    It handles JSON parsing errors gracefully and filters out invalid responses.
    """
    
    def __init__(self, *, fail_fast: bool = False, **kwargs):
        """Initialize the JSON parser.

        By default ``fail_fast`` is set to *False* so that a handful of
        malformed JSON responses do **not** crash the entire pipeline.  You
        can opt-in to strict mode by passing ``fail_fast=True``.
        """
        super().__init__(fail_fast=fail_fast, **kwargs)
        
    def run(self, data: PropertyDataset) -> PropertyDataset:
        """
        Parse raw LLM responses into Property objects.
        
        Args:
            data: PropertyDataset with properties containing raw LLM responses
            
        Returns:
            PropertyDataset with parsed and validated properties
        """
        self.log(f"Parsing {len(data.properties)} raw property responses")
        
        parsed_properties: List[Property] = []
        parse_errors = 0
        
        for prop in data.properties:
            # We only process properties that still have raw_response
            if not prop.raw_response:
                parsed_properties.append(prop)
                continue

            parsed_json = self._parse_json_response(prop.raw_response)
            if parsed_json is None:
                parse_errors += 1
                # Debug: show a snippet of the offending response to aid troubleshooting
                snippet = (prop.raw_response or "")[:200].replace("\n", " ")
                self.log(
                    f"Failed to parse JSON for property {prop.id}. Snippet: {snippet}…",
                    level="error",
                )
                self.handle_error(ValueError("Failed to parse JSON"), f"property {prop.id}")
                continue

            # The LLM might return a single property dict or {"properties": [...]} or a list
            if isinstance(parsed_json, dict) and "properties" in parsed_json:
                prop_dicts = parsed_json["properties"]
            elif isinstance(parsed_json, list):
                prop_dicts = parsed_json
            elif isinstance(parsed_json, dict):
                prop_dicts = [parsed_json]
            else:
                self.handle_error(ValueError("Unsupported JSON shape"), f"property {prop.id}")
                parse_errors += 1
                continue

            for p_dict in prop_dicts:
                try:
                    parsed_properties.append(self._to_property(p_dict, prop))
                except Exception as e:
                    parse_errors += 1
                    self.handle_error(e, f"building property from JSON for {prop.question_id}")

        self.log(f"Parsed {len(parsed_properties)} properties successfully, {parse_errors} failed")
        
        # Log to wandb if enabled
        if hasattr(self, 'use_wandb') and self.use_wandb:
            self._log_parsing_to_wandb(data.properties, parsed_properties, parse_errors)
        
        return PropertyDataset(
            conversations=data.conversations,
            properties=parsed_properties,
            clusters=data.clusters,
            model_stats=data.model_stats
        )
    
    def _parse_json_response(self, response_text: str) -> Optional[Any]:
        """
        Parse JSON response from model, handling potential formatting issues.
        
        This method migrates the parse_json_response function from post_processing.py.
        """
        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_content = response_text[json_start:json_end].strip()
            else:
                json_content = response_text.strip()
            parsed_json = json.loads(json_content)
            return parsed_json
        except Exception as e:
            return None 

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _to_property(self, p: Dict[str, Any], prop: Property) -> Property:
        """Convert a dict returned by the LLM into a Property object."""

        if len(prop.model) > 1:
            model = model_name_pass(p.get("model", "unknown"), prop.model[0], prop.model[1])
        else:
            model = prop.model

        return Property(
            id=str(uuid.uuid4()),
            question_id=prop.question_id,
            model=model,
            property_description=p.get("property_description"),
            category=p.get("category"),
            type=p.get("type"),
            impact=p.get("impact"),
            reason=p.get("reason"),
            evidence=p.get("evidence"),
            contains_errors=p.get("contains_errors"),
            unexpected_behavior=p.get("unexpected_behavior"),
            user_preference_direction=p.get("user_preference_direction"),
        ) 

    def _log_parsing_to_wandb(self, raw_properties: List[Property], parsed_properties: List[Property], parse_errors: int):
        """Log parsing results to wandb."""
        try:
            import wandb
            
            # Calculate parsing success rate
            total_raw = len(raw_properties)
            total_parsed = len(parsed_properties)
            parse_success_rate = total_parsed / total_raw if total_raw > 0 else 0
            
            # Log parsing metrics
            self.log_wandb({
                "parsing_total_properties": total_raw,
                "parsing_successful_properties": total_parsed,
                "parsing_failed_properties": parse_errors,
                "parsing_success_rate": parse_success_rate,
            })
            
            # Log a sample of parsed properties
            if parsed_properties:
                sample_size = min(100, len(parsed_properties))
                sample_properties = parsed_properties[:sample_size]
                
                property_data = []
                for prop in sample_properties:
                    property_data.append({
                        "question_id": prop.question_id,
                        "model": prop.model,
                        "property_description": prop.property_description,
                        "category": prop.category,
                        "impact": prop.impact,
                        "type": prop.type,
                        "user_preference_direction": prop.user_preference_direction,
                        "contains_errors": prop.contains_errors,
                        "unexpected_behavior": prop.unexpected_behavior,
                    })
                
                self.log_wandb({
                    "parsed_properties_sample": wandb.Table(
                        columns=["question_id", "model", "property_description", "category", "impact", "type", "user_preference_direction", "contains_errors", "unexpected_behavior"],
                        data=[[row[col] for col in ["question_id", "model", "property_description", "category", "impact", "type", "user_preference_direction", "contains_errors", "unexpected_behavior"]] 
                              for row in property_data]
                    )
                })
            
            # Log parsing error count
            if parse_errors > 0:
                self.log_wandb({
                    "parsing_error_count": parse_errors,
                })
            
        except Exception as e:
            self.log(f"Failed to log parsing to wandb: {e}", level="warning") 

def remove_things(x):
    x = x[x.find('_')+1:]
    x = x.replace("-Instruct", "")
    return x.lower()

def model_name_pass(model, model_a, model_b):
    model_a_modified_name = remove_things(model_a)
    model_b_modified_name = remove_things(model_b)
    model_modified_name = remove_things(model)
    if model == model_a or model.lower() == "model a" or model_modified_name == model_a_modified_name:
        return model_a
    if model == model_b or model.lower() == "model b" or model_modified_name == model_b_modified_name:
        return model_b
    return "unknown"