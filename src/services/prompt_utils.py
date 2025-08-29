from pathlib import Path

import yaml


class PromptUtils:
    yaml_paths = {
        "qualification_analysis": Path(__file__).parent / "prompts" / "qualification_prompt_templates.yaml",
        "sql_agent": Path(__file__).parent / "prompts" / "sql_prompt_templates.yaml",
        "validation_question": Path(__file__).parent / "prompts" / "validation_template.yaml"
    }

    @staticmethod
    def load_prompt_template(template_name: str, version: str = "stable") -> dict:
        """Load prompt template from YAML configuration file."""
        config_path = PromptUtils.yaml_paths[template_name]
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        template_config = config[template_name]
        if version == "stable":
            version = template_config["stable"]
        return template_config[version]