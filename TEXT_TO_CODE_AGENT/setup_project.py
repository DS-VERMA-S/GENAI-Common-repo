from pathlib import Path

PROJECT_NAME = "TEXT_TO_CODE_AGENT"

STRUCTURE = {
    "agent": {
        "__init__.py": "",
        "state.py": "",
        "graph.py": "",
        "nodes": {
            "__init__.py": "",
            "planner.py": "",
            "sql_generator.py": "",
            "validator.py": "",
            "formatter.py": "",
        },
    },
    "tools": {
        "__init__.py": "",
        "schema_loader.py": "",
        "sql_executor.py": "",
    },
    "prompts": {
        "planner_prompt.txt": "",
        "sql_prompt.txt": "",
    },
    "tests": {
        "__init__.py": "",
        "test_validation.py": "",
        "test_queries.py": "",
    },
    "config.yaml": "",
    "main.py": "",
    "README.md": "# Text-to-SQL Agent\n\nProduction-grade agent using LangGraph.\n",
}


def create_structure(base_path: Path, structure: dict):
    for name, content in structure.items():
        path = base_path / name

        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_structure(path, content)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(content)


def main():
    base_path = Path(PROJECT_NAME)
    base_path.mkdir(exist_ok=True)
    create_structure(base_path, STRUCTURE)
    print(f"Project '{PROJECT_NAME}' created successfully.")


if __name__ == "__main__":
    main()
