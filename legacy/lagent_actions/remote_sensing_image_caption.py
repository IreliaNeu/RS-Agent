import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


from lagent.actions.base_action import BaseAction, tool_api
from typing import Optional, Type
from lagent.actions.parser import JsonParser, BaseParser
from lagent.schema import ActionReturn, ActionStatusCode
from Image_Caption.caption_pipeline import generate_and_refine_pipeline


class RemoteSensingImageCaption(BaseAction):
    """Tool: Generate + refine remote sensing image captions and export dataset"""

    def __init__(
        self,
        description: Optional[dict] = {
            "name": "remote_sensing_image_caption",
            "description": "Generate multimodal caption dataset for remote sensing images",
            "parameters": [
                {
                    "name": "image_dir",
                    "type": "string",
                    "description": "Directory containing input images",
                },
                {
                    "name": "output_dir",
                    "type": "string",
                    "description": "Directory to save output dataset",
                },
                {
                    "name": "api_key",
                    "type": "string",
                    "description": "API key for the caption service",
                },
            ],
            "required": ["image_dir", "output_dir", "api_key"],
        },
        parser: Type[BaseParser] = JsonParser,
        enable: bool = True,
    ):
        super().__init__(description, parser, enable)

    @tool_api
    def run(self, image_dir: str, output_dir: str, api_key: str) -> ActionReturn:
        tool_return = ActionReturn(type=self.name)
        try:
            result_path = generate_and_refine_pipeline(image_dir, output_dir, api_key)
            tool_return.result = [{"type": "text", "content": f"✅ Dataset saved to: {result_path}"}]
            tool_return.state = ActionStatusCode.SUCCESS
        except Exception as e:
            tool_return.errmsg = repr(e)
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return
