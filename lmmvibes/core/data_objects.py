"""
Core data objects for LMM-Vibes pipeline.

These objects define the data contract that flows between pipeline stages.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field, validator
import numpy as np


@dataclass
class ConversationRecord:
    """A single conversation with prompt, responses, and metadata."""
    question_id: str
    prompt: str
    model: str | tuple[str, str]  # model name(s) - single string or tuple for side-by-side comparisons
    responses: str | tuple[str, str] # {model_name: response}
    scores: Dict[str, Any]     # {score_name: score_value}
    meta: Dict[str, Any] = field(default_factory=dict)  # winner, language, etc.

@dataclass
class Property:
    """An extracted behavioral property from a model response."""
    id: str
    question_id: str
    model: str
    # Parsed fields (filled by LLMJsonParser)
    property_description: Optional[str] = None
    category: Optional[str] = None
    type: Optional[str] = None  # "Context-Specific" or "General"
    impact: Optional[str] = None  # "High", "Medium", "Low"
    reason: Optional[str] = None
    evidence: Optional[str] = None
    user_preference_direction: Optional[str] = None # Capability-focused|Experience-focused|Neutral|Negative

    # Raw LLM response (captured by extractor before parsing)
    raw_response: Optional[str] = None
    contains_errors: Optional[bool] = None
    unexpected_behavior: Optional[bool] = None

    def to_dict(self):
        return asdict(self)
    
    def __post_init__(self):
        """Validate property fields after initialization."""
        # Only validate once fields are parsed / populated
        if self.type is not None and self.type not in ["Context-Specific", "General"]:
            raise ValueError(
                f"Property type must be 'Context-Specific' or 'General', got: {self.type}"
            )
        if self.impact is not None and self.impact not in ["High", "Medium", "Low"]:
            raise ValueError(
                f"Property impact must be 'High', 'Medium', or 'Low', got: {self.impact}"
            )
        if self.user_preference_direction is not None and self.user_preference_direction not in ["Capability-focused", "Experience-focused", "Neutral", "Negative"]:
            raise ValueError(
                f"User preference direction must be 'Capability-focused', 'Experience-focused', 'Neutral', or 'Negative', got: {self.user_preference_direction}"
            )

@dataclass
class Cluster:
    """A cluster of properties."""
    id: str # fine cluster id
    label: str # fine cluster label
    size: int # fine cluster size
    parent_id: str | None = None # coarse cluster id
    parent_label: str | None = None # coarse cluster label
    property_descriptions: List[str] = field(default_factory=list) # property descriptions in the cluster
    question_ids: List[str] = field(default_factory=list) # ids of the conversations in the cluster

    def to_dict(self):
        return asdict(self)

@dataclass
class PropertyDataset:
    """
    Container for all data flowing through the pipeline.
    
    This is the single data contract between all pipeline stages.
    """
    conversations: List[ConversationRecord] = field(default_factory=list)
    all_models: List[str] = field(default_factory=list)
    properties: List[Property] = field(default_factory=list)
    clusters: List[Cluster] = field(default_factory=list)
    model_stats: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, method: str = "side_by_side") -> "PropertyDataset":
        """
        Create PropertyDataset from existing DataFrame formats.
        
        Args:
            df: Input DataFrame with conversation data
            method: "side_by_side" for comparison data, "single_model" for single responses
            
        Returns:
            PropertyDataset with populated conversations
        """
        conversations = []
        
        if method == "side_by_side":
            all_models = list(set(df["model_a"].unique().tolist() + df["model_b"].unique().tolist()))
            # Expected columns: question_id, prompt, model_a, model_b, 
            # model_a_response, model_b_response, winner, etc.
            for _, row in df.iterrows():
                conversation = ConversationRecord(
                    question_id=str(row.get('question_id', row.name)),
                    prompt=str(row.get('prompt', row.get('user_prompt', ''))),
                    model=(row.get('model_a', 'model_a'), row.get('model_b', 'model_b')),
                    responses=(row.get('model_a_response', ''), row.get('model_b_response', '')),
                    scores=row.get('score', {}),
                    meta={k: v for k, v in row.items() 
                          if k not in ['question_id', 'prompt', 'user_prompt', 'model_a', 'model_b', 
                                     'model_a_response', 'model_b_response', 'score']}
                )
                conversations.append(conversation)
                
        elif method == "single_model":
            all_models = df["model"].unique().tolist()
            # Expected columns: question_id, prompt, model, model_response, score, etc.
            for _, row in df.iterrows():
                conversation = ConversationRecord(
                    question_id=str(row.get('question_id', row.name)),
                    prompt=str(row.get('prompt', row.get('user_prompt', ''))),
                    model=str(row.get('model', 'model')),
                    responses=str(row.get('model_response', '')),
                    scores={
                        'score': row.get('score', 0)
                    },
                    meta={k: v for k, v in row.items() 
                          if k not in ['question_id', 'prompt', 'user_prompt', 'model', 'model_response', 'score']}
                )
                conversations.append(conversation)
        else:
            raise ValueError(f"Unknown method: {method}. Must be 'side_by_side' or 'single_model'")
            
        return cls(conversations=conversations, all_models=all_models)
    
    def to_dataframe(self, type: str = "all") -> pd.DataFrame:
        """
        Convert PropertyDataset back to DataFrame format.
        
        Returns:
            DataFrame with original data plus extracted properties and clusters
        """

        assert type in ["base", "properties", "clusters", "all"], f"Invalid type: {type}. Must be 'all' or 'base'"
        # Start with conversation data
        rows = []
        for conv in self.conversations:
            if len(conv.model) == 1:
              
                base_row = {
                    'question_id': conv.question_id,
                    'prompt': conv.prompt,
                    'model': conv.model,
                    'responses': conv.responses,
                    **conv.scores,
                    **conv.meta
                }
            else:
                base_row = {
                    'question_id': conv.question_id,
                    'prompt': conv.prompt,
                    'model_a': conv.model[0],
                    'model_b': conv.model[1],
                    'model_a_response': conv.responses[0],
                    'model_b_response': conv.responses[1],
                    **conv.scores,
                    **conv.meta
                }

            rows.append(base_row)
        
        df = pd.DataFrame(rows)
        
        # Add properties if they exist
        if self.properties and type in ["all", "properties", "clusters"]:
            # Create a mapping from (question_id, model) to properties
            prop_map = {}
            for prop in self.properties:
                key = (prop.question_id, prop.model)
                if key not in prop_map:
                    prop_map[key] = []
                prop_map[key].append(prop)
            
            # Add property columns
            property_cols = [
                'property_description', 'category', 'type', 'impact', 
                'reason', 'evidence', 'contains_errors', 'unexpected_behavior', 'user_preference_direction'
            ]
            
            # create property df
            prop_df = pd.DataFrame([p.to_dict() for p in self.properties])
            print("len df before merge ", len(df))
            print("prop_df ", prop_df.columns)
            df = df.merge(prop_df, on="question_id", how="left")
            print("len df after merge ", len(df))
            print("df ", df.columns)

        if self.clusters and type in ["all", "clusters"]:
            # create cluster df
            cluster_df = pd.DataFrame([c.to_dict() for c in self.clusters])
            cluster_df.rename(columns={"id": "fine_cluster_id", "label": "fine_cluster_label", "size": "fine_cluster_size", "parent_id": "coarse_cluster_id", "parent_label": "coarse_cluster_label", "property_descriptions": "property_description"}, inplace=True)
            cluster_df = cluster_df.explode("property_description")
            print("len df before merge ", len(df))
            print("cluster_df ", cluster_df.columns)
            print("df ", df.columns)
            df = df.merge(cluster_df, on=["property_description"], how="left")
            print("len df after merge ", len(df))
            print("df ", df.columns)
        
        return df
    
    def add_property(self, property: Property):
        """Add a property to the dataset."""
        self.properties.append(property)
    
    def get_properties_for_model(self, model: str) -> List[Property]:
        """Get all properties for a specific model."""
        return [p for p in self.properties if p.model == model]
    
    def get_properties_for_question(self, question_id: str) -> List[Property]:
        """Get all properties for a specific question."""
        return [p for p in self.properties if p.question_id == question_id]

    def _json_safe(self, obj: Any):
        """Recursively convert *obj* into JSON-safe types (lists, dicts, ints, floats, strings, bool, None)."""
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (list, tuple, set)):
            return [self._json_safe(o) for o in obj]
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}
        # Fallback – use string representation
        return str(obj)

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Convert the whole dataset into a JSON-serialisable dict."""
        return {
            "conversations": [self._json_safe(asdict(conv)) for conv in self.conversations],
            "properties": [self._json_safe(asdict(prop)) for prop in self.properties],
            "clusters": self._json_safe(self.clusters),
            "model_stats": self._json_safe(self.model_stats),
        }

    # ------------------------------------------------------------------
    # 📝 Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str, format: str = "json") -> None:
        """Save the dataset to *path* in either ``json``, ``dataframe``, ``parquet`` or ``pickle`` format.

        The JSON variant produces a fully human-readable file while the pickle
        variant preserves the exact Python objects.
        """
        import json, pickle, os

        fmt = format.lower()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_serializable_dict(), f, ensure_ascii=False, indent=2)
        elif fmt == "dataframe":
            self.to_dataframe().to_json(path, orient="records", lines=True)
        elif fmt == "parquet":
            self.to_dataframe().to_parquet(path)
        elif fmt in {"pkl", "pickle"}:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'.")

    @classmethod
    def load(cls, path: str, format: str = "json") -> "PropertyDataset":
        """Load a dataset previously saved with :py:meth:`save`."""
        import json, pickle

        fmt = format.lower()
        if fmt == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            conversations = [ConversationRecord(**conv) for conv in data.get("conversations", [])]
            properties = [Property(**prop) for prop in data.get("properties", [])]
            clusters = data.get("clusters", {})
            model_stats = data.get("model_stats", {})
            return cls(conversations=conversations, properties=properties, clusters=clusters, model_stats=model_stats)
        elif fmt in {"pkl", "pickle"}:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if not isinstance(obj, cls):
                raise TypeError("Pickle file does not contain a PropertyDataset object")
            return obj
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'.") 