import json
from llm_connect import LLMConnector, OpenAILib, OpenAIChat
import random
from tqdm import tqdm
from joblib import delayed, Parallel
import copy
import sys
import time
import os
import run_experiment as run_exper

thread_count = 50
unified_fewshot: bool = False
uniform_game_id_sampling = True

def get_finetune(connector: OpenAIChat, sample: dict) -> dict:
    request = run_exper.sample_to_request(sample)
    response = run_exper.sample_to_response(sample)
    connector.inject(json.dumps(request), json.dumps(response))
    return connector.as_fine_tuning_example()

if __name__ == '__main__':
    with open(run_exper.prompt_filename, 'r') as f:
        prompt_template = f.read()
    samples_per_file = int(sys.argv[1])
    dataset_filenames = sys.argv[2:]
    base_dataset_names = '-'.join([(os.path.splitext(os.path.basename(dataset_filename))[0]).split('_')[0] for dataset_filename in dataset_filenames])
    out_filename = os.path.join(f'finetune_dataset', f's{samples_per_file}_per_{base_dataset_names}_{int(time.time())}.json')
    json_samples: list[dict] = []
    for dataset_filename in dataset_filenames:
        with open(dataset_filename, 'r') as f:
            dataset = json.load(f)
        game_name: str = dataset['name']
        game_desc: str = dataset['description']
        samples: list[dict] = run_exper.prepare_samples(dataset['samples'], samples_per_file, random.Random(), uniform_game_id_sampling)
        print(f'{len(samples)} samples from {dataset_filename}')
        fewshot: list[dict]
        # choose randomly, equal valid and invalid
        # possibly try to have different moves
        prompt = run_exper.fill_prompt(prompt_template, game_name, game_desc)
        connector = OpenAIChat(OpenAIChat.OpenAIModel.GPT_4O_mini)
        if unified_fewshot:
            samples, fewshot = run_exper.fewshot_split(samples, run_exper.fewshot_seed)
            run_exper.initiate_conversation(connector, prompt, fewshot)
        fewshot_rnd = random.Random(run_exper.fewshot_seed) # only used for non-unified
        for i, sample in enumerate(samples):
            connector_copy = connector.copy()
            if not unified_fewshot:
                fewshot = run_exper.get_single_sample_fewshot(samples, i, fewshot_rnd.randint(0, 1000000))
                run_exper.initiate_conversation(connector_copy, prompt, fewshot)
            json_samples.append(get_finetune(connector_copy, sample))
    random.shuffle(json_samples)
    print(f"total sample count: {len(json_samples)}")
    print(f"saving as: {out_filename}")
    with open(f'{out_filename}.jsonl', 'a') as outfile:
        for json_sample in json_samples:
            json.dump(json_sample, outfile)
            outfile.write('\n')
    # results = {
    #     'llm': connector.model_name,
    #     'dataset': dataset_filename,
    #     'initial_conversation': copy.deepcopy(connector.chat_log),
    #     'samples': result_samples
    # }
    # results['sample_count'] = len(result_samples)
    # results['legal_accuracy'] = sum([1 if sample['legal'] == sample['pred_legal'] else 0 for sample in results['samples']])
    # results['next_state_accuracy'] = sum([1 if sample['next_state'] == sample['pred_next_state'] else 0 for sample in results['samples']])
    # results['legal_next_state_accuracy'] = sum([1 if sample['next_state'] == sample['pred_next_state'] else 0 for sample in results['samples'] if sample['legal']])
    # results['legal_count'] = sum([1 for sample in results['samples'] if sample['legal']])

    # base_dataset_name = os.path.splitext(os.path.basename(dataset_filename))[0]
    # out_filename = os.path.join(f'results', f'{base_dataset_name}_{int(time.time())}.json')
    # with open(out_filename, 'w') as f:
    #     json.dump(results, f, indent=4)
