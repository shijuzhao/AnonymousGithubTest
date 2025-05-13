import os
import logging
logging.basicConfig(level=logging.INFO)
class Configs:
    cwd = '~/InfoBlend/benchmarks_infoblend/test_type/prob_attn' # Current working directory
    all_datasets = ['example']
    all_models = ['benchmarks_infoblend/models/llava-v1.6-vicuna-7b-hf', 'benchmarks_infoblend/models/InternVL2_5-8B']

    def __init__(self, dataset, model):

        self.dataset = dataset
        self.model = model
        self._verify_init_args()

        # Create parent folders for results
        self.result_folder = os.path.join('results', dataset, model.split('/')[-1])
        os.makedirs(self.result_folder, exist_ok=True)

        # Other configs
        self.seed = 42

    def _verify_init_args(self):
        assert self.model in self.all_models
        assert self.dataset in self.all_datasets