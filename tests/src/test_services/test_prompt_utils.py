import os
import sys
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from services.prompt_utils import PromptUtils


class TestPromptUtils:
    
    @pytest.fixture
    def sample_yaml_content(self):
        """Sample YAML content for testing."""
        return {
                "qualification_analysis": {
                    "stable": "v0",
                    "v0": {
                        "name": "Test Template v0",
                        "input_variables": ["question", "sql_data", "rules", "language"],
                        "template": "Test template content for v0"
                    },
                    "v1": {
                        "name": "Test Template v1", 
                        "input_variables": ["question", "sql_data", "rules", "language", "extra"],
                        "template": "Test template content for v1"
                    }
                }
        }

    def test_load_prompt_template_with_stable_version(self, sample_yaml_content):
        """Test loading prompt template with stable version."""
        yaml_content = yaml.dump(sample_yaml_content)
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            result = PromptUtils.load_prompt_template("qualification_analysis", "stable")
            
        expected = sample_yaml_content["qualification_analysis"]["v0"]
        assert result == expected
        assert result["name"] == "Test Template v0"

    def test_load_prompt_template_with_specific_version(self, sample_yaml_content):
        """Test loading prompt template with specific version."""
        yaml_content = yaml.dump(sample_yaml_content)
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            result = PromptUtils.load_prompt_template("qualification_analysis", "v1")
            
        expected = sample_yaml_content["qualification_analysis"]["v1"]
        assert result == expected
        assert result["name"] == "Test Template v1"

    def test_load_prompt_template_default_version(self, sample_yaml_content):
        """Test loading prompt template with default version (stable)."""
        yaml_content = yaml.dump(sample_yaml_content)
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            result = PromptUtils.load_prompt_template("qualification_analysis")
            
        # Should default to stable version (v0)
        expected = sample_yaml_content["qualification_analysis"]["v0"]
        assert result == expected

    def test_yaml_paths_configuration(self):
        """Test that yaml_paths is properly configured."""
        assert "qualification_analysis" in PromptUtils.yaml_paths
        assert isinstance(PromptUtils.yaml_paths["qualification_analysis"], Path)
        assert str(PromptUtils.yaml_paths["qualification_analysis"]).endswith("qualification_prompt_templates.yaml")

    def test_load_prompt_template_missing_template_name(self, sample_yaml_content):
        """Test behavior when template name doesn't exist."""
        yaml_content = yaml.dump(sample_yaml_content)
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with pytest.raises(KeyError):
                PromptUtils.load_prompt_template("nonexistent_template")

    def test_load_prompt_template_missing_version(self, sample_yaml_content):
        """Test behavior when requested version doesn't exist."""
        yaml_content = yaml.dump(sample_yaml_content)
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with pytest.raises(KeyError):
                PromptUtils.load_prompt_template("qualification_analysis", "v99")

    def test_load_prompt_template_file_not_found(self):
        """Test behavior when YAML file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                PromptUtils.load_prompt_template("qualification_analysis")

    def test_load_prompt_template_invalid_yaml(self):
        """Test behavior when YAML content is invalid."""
        invalid_yaml = "invalid: yaml: content: ["
        
        with patch("builtins.open", mock_open(read_data=invalid_yaml)):
            with pytest.raises(yaml.YAMLError):
                PromptUtils.load_prompt_template("qualification_analysis")

    def test_load_prompt_template_missing_stable_reference(self):
        """Test behavior when stable reference is missing."""
        yaml_content_no_stable = {
            "prompt_templates": {
                "qualification_analysis": {
                    "v0": {
                        "name": "Test Template v0",
                        "template": "Test content"
                    }
                }
            }
        }
        yaml_content = yaml.dump(yaml_content_no_stable)
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with pytest.raises(KeyError):
                PromptUtils.load_prompt_template("qualification_analysis", "stable")