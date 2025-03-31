from benchmarks_infoblend.data.data_set import DataSet
from typing import Dict
import logging
from transformers import AutoTokenizer
import pandas as pd
logger = logging.getLogger(__name__)

class Example(DataSet):
    def load_data_from_hf(self):
        return pd.DataFrame({
            'images': [[
                'benchmarks_infoblend/data/example_images/IMAGE#EIFFEL2025.jpeg', 
                'benchmarks_infoblend/data/example_images/IMAGE#LOUVRE2025.jpeg'
            ]],
            'input': ["I'm planning a trip to Paris and I've found two images online. This one is of the Eiffel Tower <image>, and this one is of the Louvre Museum <image>. Which one do you think has a more iconic design?"],
            'answers': [['Both are iconic in their own right, but the Eiffel Tower is often considered more iconic due to its unique structure and global recognition.']],
            'length' : [100] # dummy value
        })
            
    def _split_docs(self, row: Dict) -> Dict:
        return row

    def _append_input(self, row: Dict) -> Dict:
        row['input'] = "<s>USER: \n\nI'm planning a trip to Paris and I've found two images online. This one is of the Eiffel Tower <image>, and this one is of the Louvre Museum <image>. Which one do you think has a more iconic design?" \
                        "</s><s>ASSISTANT: \n\n"
        return row

    def _append_system_prompt(self, row: Dict) -> Dict:
        row['system_prompt'] = "<s>SYSTEM: \n\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n</s>"
        return row
    
    def _custom_process_data(self) -> None:
        pass