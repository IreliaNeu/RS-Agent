import copy
import io
from contextlib import redirect_stdout
from typing import Any, Optional, Type

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode
from pathlib import Path
from PIL import Image

class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(
            self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)


class ConvertToMultimodalDataset(BaseAction):
    """
    Converts a dataset of images and labels to a multimodal dataset.
    将变化检测图像对批量转换为多模态数据（图像 + 文字描述）。

    """

    def __init__(self,
                 answer_symbol: Optional[str] = None,
                 answer_expr: Optional[str] = 'solution()',
                 answer_from_stdout: bool = False,
                 timeout: int = 5000,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description, parser, enable)
        self.answer_symbol = answer_symbol
        self.answer_expr = answer_expr
        self.answer_from_stdout = answer_from_stdout
        self.timeout = timeout


    @tool_api
    def run(self,command: str) -> ActionReturn:
        """用来执行Python代码。代码必须是一个函数，函数名必须得是 'solution'，代码对应你的思考过程。代码实例格式如下：

        ```python
        # import 依赖包
        import xxx
        def solution():
            # 初始化一些变量
            variable_names_with_real_meaning = xxx
            # 步骤一
            mid_variable = func(variable_names_with_real_meaning)
            # 步骤 x
            mid_variable = func(mid_variable)
            # 最后结果
            final_answer =  func(mid_variable)
            return final_answer
        ```

        Note:
        In addition to the commonly used python dependency packages, you can use a custom library 'Change_Perception' from tools import Change_Perception(), which contains the following function:
        1. **`Change_Perception.convert_to_multimodal(input_dir_A, input_dir_B, output_dir)`**:
           - **Parameters**:
             - `input_dir_A`: Path to the first images file.
             - `input_dir_B`: Path to the second images file.
             - `output_dir`: Path to save the output_text.
           - **Returns**:
             - A file: text.txt(which contains the description of the changes between the two images.) 
        NOTE:  The code of `Action Input` must be placed inside `def solution():`!  
            For example:
            When a user wishes to convert a dataset into multimodal data,"Action Input" should be as follows:
            ```python
            def solution():
                from tools import Change_Perception
                import os

                #Set the image directory path (must be the decompressed folder)
                input_dir_A = r'.\tmp_dir\image_file_A'
                input_dir_B = r'.\tmp_dir\image_file_B'
                output_dir = r'.\tmp_dir\output_text'
                # #Verify whether the path exists (to prevent path errors)
                assert os.path.exists(input_dir_A), f"路径不存在: {input_dir_A}"
                assert os.path.exists(input_dir_B), f"路径不存在: {input_dir_B}"
                os.makedirs(output_dir, exist_ok=True)

                # Instantiate the model and execute the conversion function
                model = Change_Perception()
                summary = model.convert_to_multimodal(
                    input_dir_A=input_dir_A,
                    input_dir_B=input_dir_B,
                    output_dir=output_dir
                )
                return summary

        Args:
            command (:class:`str`): Python code snippet。

        Returns:
            ActionReturn: 包含脚本执行结果或异常信息
        """
        from func_timeout import FunctionTimedOut, func_set_timeout
        self.runtime = GenericRuntime()
        try:
            tool_return = func_set_timeout(self.timeout)(self._call)(command)
        except FunctionTimedOut as e:
            tool_return = ActionReturn(type=self.name)
            tool_return.errmsg = repr(e)
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return


    def _call(self, command: str) -> ActionReturn:
        tool_return = ActionReturn(type=self.name)
        print('RUN Command:', command)
        try:
                        # 清理 Markdown 代码块包裹
            if '```python' in command:
                command = command.split('```python')[1].split('```')[0]
            elif '```' in command:
                command = command.split('```')[1].split('```')[0]

            tool_return.args = dict(text='```python\n' + command + '\n```')
            command_lines = command.split('\n')

            if self.answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    self.runtime.exec_code('\n'.join(command_lines))
                program_io.seek(0)
                res = program_io.readlines()[-1]
            elif self.answer_symbol:
                self.runtime.exec_code('\n'.join(command_lines))
                res = self.runtime._global_vars[self.answer_symbol]
            elif self.answer_expr:
                self.runtime.exec_code('\n'.join(command_lines))
                res = self.runtime.eval_code(self.answer_expr)
            else:
                self.runtime.exec_code('\n'.join(command_lines[:-1]))
                res = self.runtime.eval_code(command_lines[-1])

            tool_return.result = [dict(type='text', content=str(res))]
            tool_return.state = ActionStatusCode.SUCCESS
        except Exception as e:
            tool_return.errmsg = repr(e)
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return