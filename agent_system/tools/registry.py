from .wrappers import (
    train_whisper_tool,
    train_llm_tool,
    make_model_zip_tool,
    analyze_dataset_tool,
    _strict_schema
)
from .schema import TOOL_SCHEMAS

class ToolRegistry:
    def __init__(self):
        self.tools = {
            "train_whisper": train_whisper_tool,
            "train_llm": train_llm_tool,
            "make_model_zip": make_model_zip_tool,
            "analyze_dataset": analyze_dataset_tool
        }
        self.schemas = TOOL_SCHEMAS

    def get_registered_tools(self) -> list[str]:
        return list(self.tools.keys())

    def get_tool_schema(self, tool_name: str) -> dict | None:
        """Returns the strictly defined schema for a given tool."""
        schema = self.schemas.get(tool_name)
        if schema:
            return schema.to_dict()
        return None

    def get_all_schemas(self) -> dict:
        """Returns a dict of all available tool schemas."""
        return {name: schema.to_dict() for name, schema in self.schemas.items()}

    async def execute(self, tool_name: str, **kwargs) -> dict:
        """
        Executes a tool asynchronously and guarantees the strict schema return.
        """
        from .security import SecurityManager, SecurityException

        if tool_name not in self.tools:
            from .wrappers import _strict_schema
            return _strict_schema(
                status_code=1,
                stdout="",
                stderr=f"Tool '{tool_name}' not found in registry.",
                metrics={}
            )
            
        try:
            # Sanitize kwargs against shell injections
            safe_kwargs = SecurityManager.sanitize_tool_kwargs(tool_name, kwargs)
            
            tool_func = self.tools[tool_name]
            # The tool_func is expected to return the strict schema dict
            result = await tool_func(**safe_kwargs)
            return result
        except SecurityException as se:
            from .wrappers import _strict_schema
            return _strict_schema(
                status_code=1,
                stdout="",
                stderr=f"Security exception: {str(se)}",
                metrics={}
            )
        except Exception as e:
            import traceback
            return _strict_schema(
                status_code=1,
                stdout="",
                stderr=f"Tool wrapper exception: {str(e)}\n{traceback.format_exc()}",
                metrics={}
            )
