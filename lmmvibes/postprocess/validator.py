"""
Property validation stage.

This stage validates and cleans extracted properties.
"""

from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset, Property
from ..core.mixins import LoggingMixin


class PropertyValidator(LoggingMixin, PipelineStage):
    """
    Validate and clean extracted properties.
    
    This stage ensures that all properties have valid data and removes
    any properties that don't meet quality criteria.
    """
    
    def __init__(self, **kwargs):
        """Initialize the property validator."""
        super().__init__(**kwargs)
        
    def run(self, data: PropertyDataset) -> PropertyDataset:
        """
        Validate and clean properties.
        
        Args:
            data: PropertyDataset with properties to validate
            
        Returns:
            PropertyDataset with validated properties
        """
        self.log(f"Validating {len(data.properties)} properties")
        
        # Print diagnostic information about models before validation
        print(f"\n🔍 Property validation diagnostic:")
        print(f"   • Input dataset has {len(data.all_models)} models: {sorted(data.all_models)}")
        print(f"   • Input properties: {len(data.properties)}")
        
        # Count properties per model before validation
        model_property_counts = {}
        for prop in data.properties:
            if isinstance(prop.model, str):
                model_property_counts[prop.model] = model_property_counts.get(prop.model, 0) + 1
            elif isinstance(prop.model, list):
                for model in prop.model:
                    model_property_counts[model] = model_property_counts.get(model, 0) + 1
        
        print(f"   • Properties per model before validation:")
        for model in sorted(data.all_models):
            count = model_property_counts.get(model, 0)
            print(f"     - {model}: {count} properties")
        print()
        
        valid_properties = []
        for prop in data.properties:
            if self._is_valid_property(prop):
                valid_properties.append(prop)
                
        self.log(f"Kept {len(valid_properties)} valid properties")
        
        # Count properties per model after validation
        valid_model_property_counts = {}
        for prop in valid_properties:
            if isinstance(prop.model, str):
                valid_model_property_counts[prop.model] = valid_model_property_counts.get(prop.model, 0) + 1
            elif isinstance(prop.model, list):
                for model in prop.model:
                    valid_model_property_counts[model] = valid_model_property_counts.get(model, 0) + 1
        
        print(f"   • Properties per model after validation:")
        for model in sorted(data.all_models):
            count = valid_model_property_counts.get(model, 0)
            print(f"     - {model}: {count} properties")
            
            # Show if model was completely filtered out
            if count == 0 and model_property_counts.get(model, 0) > 0:
                print(f"       ⚠️  All properties for this model were filtered out!")
            elif count < model_property_counts.get(model, 0):
                original_count = model_property_counts.get(model, 0)
                filtered_out = original_count - count
                print(f"       ⚠️  {filtered_out} properties filtered out during validation ({count}/{original_count} kept)")
        print()
        
        # Summary statistics
        total_original = sum(model_property_counts.values())
        total_valid = sum(valid_model_property_counts.values())
        total_filtered = total_original - total_valid
        print(f"   • Validation Summary:")
        print(f"     - Total properties before validation: {total_original}")
        print(f"     - Total properties after validation: {total_valid}")
        print(f"     - Total properties filtered out: {total_filtered}")
        print(f"     - Validation success rate: {total_valid/total_original*100:.1f}%")
        print()
        
        # Check for 0 valid properties and provide helpful error message
        if len(valid_properties) == 0:
            raise RuntimeError(
                "\n" + "="*60 + "\n"
                "ERROR: 0 valid properties after validation!\n"
                "="*60 + "\n"
                "This typically indicates one of the following issues:\n\n"
                "1. JSON PARSING FAILURES:\n"
                "   - The LLM is returning natural language instead of JSON\n"
                "   - Check the logs above for 'Failed to parse JSON' errors\n"
                "   - Verify your OpenAI API key and quota limits\n\n"
                "2. SYSTEM PROMPT ISSUES:\n"
                "   - The system prompt may not be suitable for your data format\n"
                "   - Try a different system_prompt parameter\n\n"
                "3. DATA FORMAT PROBLEMS:\n"
                "   - Input conversations may be malformed or empty\n"
                "   - Check that 'model_response' and 'prompt' columns contain valid data\n\n"
                "4. API/MODEL CONFIGURATION:\n"
                "   - OpenAI API connectivity issues\n"
                "   - Model configuration problems\n\n"
                "DEBUGGING STEPS:\n"
                "- Check for 'Failed to parse JSON' errors in the logs above\n"
                "- Verify your OpenAI API key: export OPENAI_API_KEY=your_key\n"
                "- Try with a smaller sample_size to test\n"
                "- Examine your input data format and content\n"
                "- Consider using a different system_prompt\n"
                "="*60
            )
        
        return PropertyDataset(
            conversations=data.conversations,
            all_models=data.all_models,
            properties=valid_properties,
            clusters=data.clusters,
            model_stats=data.model_stats
        )
    
    def _is_valid_property(self, prop: Property) -> bool:
        """Check if a property is valid."""
        # Basic validation - property description should exist and not be empty
        return bool(prop.property_description and prop.property_description.strip()) 