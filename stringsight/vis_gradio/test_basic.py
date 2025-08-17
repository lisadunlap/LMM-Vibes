"""
Basic test script for LMM-Vibes Gradio visualization.

This script tests basic functionality without requiring actual data.
"""

import sys
import pandas as pd
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from stringsight.vis_gradio import launch_app, create_app
        print("✅ Main imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        from stringsight.vis_gradio.data_loader import DataCache, validate_results_directory
        from stringsight.vis_gradio.utils import extract_quality_score, compute_model_rankings
        from stringsight.vis_gradio.app import create_app
        print("✅ Module imports successful")
    except ImportError as e:
        print(f"❌ Module import error: {e}")
        return False
    
    return True


def test_utility_functions():
    """Test utility functions with sample data."""
    print("Testing utility functions...")
    
    from stringsight.vis_gradio.utils import (
        extract_quality_score, 
        compute_model_rankings,
        create_model_summary_card,
        format_cluster_dataframe
    )
    
    # Test extract_quality_score
    assert extract_quality_score(0.5) == 0.5
    assert extract_quality_score({"score": 0.7}) == 0.7
    assert extract_quality_score(None) == 0.0
    print("✅ extract_quality_score works")
    
    # Test compute_model_rankings with sample data
    sample_stats = {
        "model_a": {
            "fine": [
                {"score": 0.8, "property_description": "test"},
                {"score": 0.6, "property_description": "test2"}
            ]
        },
        "model_b": {
            "fine": [
                {"score": 0.9, "property_description": "test3"}
            ]
        }
    }
    
    rankings = compute_model_rankings(sample_stats)
    assert len(rankings) == 2
    assert rankings[0][0] == "model_b"  # Higher average score
    print("✅ compute_model_rankings works")
    
    # Test HTML generation
    html = create_model_summary_card("test_model", sample_stats["model_a"])
    assert "test_model" in html
    assert "div" in html
    print("✅ create_model_summary_card works")
    
    # Test DataFrame formatting
    sample_df = pd.DataFrame({
        "model": ["model_a", "model_b"],
        "property_description": ["desc1", "desc2"],
        "fine_cluster_id": [1, 2],
        "fine_cluster_label": ["label1", "label2"],
        "score": [0.5, 0.7]
    })
    
    formatted = format_cluster_dataframe(sample_df)
    assert len(formatted) == 2
    print("✅ format_cluster_dataframe works")
    
    return True


def test_data_cache():
    """Test the data caching functionality."""
    print("Testing data cache...")
    
    from stringsight.vis_gradio.data_loader import DataCache
    
    # Test basic cache operations
    DataCache.set("test_key", "test_value")
    assert DataCache.get("test_key") == "test_value"
    assert DataCache.get("nonexistent") is None
    
    DataCache.clear()
    assert DataCache.get("test_key") is None
    
    print("✅ DataCache works")
    return True


def test_app_creation():
    """Test that the Gradio app can be created."""
    print("Testing app creation...")
    
    try:
        from stringsight.vis_gradio.app import create_app
        app = create_app()
        print("✅ Gradio app creation successful")
        return True
    except Exception as e:
        print(f"❌ App creation error: {e}")
        return False


def test_validation():
    """Test directory validation."""
    print("Testing validation functions...")
    
            from stringsight.vis_gradio.data_loader import validate_results_directory
    
    # Test with non-existent directory
    is_valid, msg = validate_results_directory("/nonexistent/path")
    assert not is_valid
    assert "does not exist" in msg
    
    # Test with empty string
    is_valid, msg = validate_results_directory("")
    assert not is_valid
    assert "provide a results directory" in msg
    
    print("✅ Validation functions work")
    return True


def main():
    """Run all tests."""
    print("🧪 Running LMM-Vibes Gradio Visualization Tests\n")
    
    tests = [
        test_imports,
        test_data_cache,
        test_utility_functions,
        test_validation,
        test_app_creation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The Gradio visualization is ready to use.")
        print("\nTo launch the app:")
        print("  python -m stringsight.vis_gradio.launcher --results_dir /path/to/results")
        print("  or")
        print("  from stringsight.vis_gradio import launch_app; launch_app()")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 