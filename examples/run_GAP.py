import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
from easyjailbreak.attacker.GAP_Schwartz_2024 import GAP
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel
from easyjailbreak.models.openai_model import OpenaiModel
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.getcwd())
dataset_name_or_path = '../easyjailbreak/datasets/data/advbench_subset.json'
dataset = JailbreakDataset(local_file_type='json', dataset=dataset_name_or_path)


openai_key = os.environ["OPENAI_KEY"]

# attack_model = OpenaiModel(model_name=attack_model_name, api_keys=openai_key)
# attack_model = MistralModel(model_name="mistral.mistral-large-2407-v1:0")
attack_model_name = 'vicuna_v1.1'
attack_model_path = 'lmsys/vicuna-13b-v1.5'
attack_model = from_pretrained(attack_model_path, model_name=attack_model_name)


eval_model_name = 'gpt-4'
eval_model = OpenaiModel(model_name=eval_model_name, api_keys=openai_key)

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map='auto')
# target_model = HuggingfaceModel(model_name='llama-3', model=model, tokenizer=tokenizer)
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto')
# target_model = HuggingfaceModel(model_name='llama-2', model=model, tokenizer=tokenizer)
target_model = OpenaiModel(model_name='gpt-3.5-turbo', api_keys=openai_key)

dataset_name = 'GAP_Atk_Vic_Evl_GPT4_Tgt_GPT35_D_AdvBench_all'
attacker = GAP(attack_model=attack_model,
               target_model=target_model,
               eval_model=eval_model,
               jailbreak_datasets=dataset,
               tree_width=10,
               tree_depth=10,
               root_num=1,
               branching_factor=4,
               keep_last_n=3,
               max_n_attack_attempts=5,
               template_file='../easyjailbreak/seed/seed_template.json',
               logging_filename=dataset_name)
eval_result = attacker.attack(save_path="{}_result.jsonl".format(dataset_name))