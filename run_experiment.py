import json
from llm_connect import LLMConnector, OpenAILib, OpenAIChat, DeepSeekChat, GeminiChat
import random
from tqdm import tqdm
from joblib import delayed, Parallel
import copy
import sys
import time
import os

prompt_filename = 'prompt.txt'
fewshot_seed = 42
fewshot_size = 4
max_fail_count = 3
thread_count = 50
unified_fewshot: bool = False
unifrom_game_id_sampling: bool = True

def extract_response(response: str) -> tuple[dict[str, str], bool]:
    if '```json' in response: # deepseek
        response = response.replace('```json', '')
        response = response.replace('```', '')
    response_obj: dict[str, str] = json.loads(response)
    tags = ['state', 'action', 'thinking', 'legal', 'next_state']
    valid = all(tag in response_obj for tag in tags)
    return dict([(tag, response_obj[tag] if tag in response_obj else '<NONE>') for tag in tags]), valid

def ask_until_fail(request: str, connector: LLMConnector) -> dict[str, str]:
    fail_count = 0
    response = None
    response_txt: str = ''
    while True:
        try:
            response_txt = connector.ask(request)
            response, valid = extract_response(response_txt)
            if not valid:
                raise Exception(f"tags missing from reponse. existing tags: {response.keys()}")
            return response
        except Exception as e:
            print(f"Response failed: {e}, response: {response_txt}\n-------------------\n\n{len(request)}\n{request[-600:]}\n---\n")
        fail_count += 1
        if fail_count == max_fail_count:
            print(f"LLM response failure {max_fail_count} times, (not) aborted")

def fewshot_split(samples: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    rand = random.Random(seed)
    fewshot_indices = rand.sample(range(len(samples)), fewshot_size)
    fewshot_samples = [samples[i] for i in fewshot_indices]
    samples = [samples[i] for i in range(len(samples)) if not i in fewshot_indices]
    return samples, fewshot_samples

def get_single_sample_fewshot(samples: list[dict], this_sample_id: int, seed: int):
    rand = random.Random(seed)
    fewshot_indices = rand.sample([i for i in range(len(samples)) if i != this_sample_id], fewshot_size)
    fewshot_samples = [samples[i] for i in fewshot_indices]
    return fewshot_samples

def sample_to_response(sample: dict) -> dict:
    response: dict[str, str] = {
        "state": sample['current_state_view'],
        "action": sample['action'],
        "thinking": sample['summary'],
        "legal": sample['is_valid'],
        "next_state": sample['next_state_view']
    }
    response['thinking'] = response['thinking'].replace('\u001b[91m', '').replace('\u001b[92m', '').replace('\u001b[0m', '')
    return response

def sample_to_request(sample: dict) -> dict:
    request = {
        "state": sample['current_state_view'],
        "action": sample['action']
    }
    return request

def initiate_conversation(connector: LLMConnector, prompt: str, fewshot_samples) -> None:
    connector.inject(prompt, '...')
    for sample in fewshot_samples:
        request = sample_to_request(sample)
        response = sample_to_response(sample)
        connector.inject(json.dumps(request), json.dumps(response))

def fill_prompt(prompt_template: str, game_name: str, game_desc: str) -> str:
    prompt = prompt_template.replace('{game_name}', game_name)\
                            .replace('{game_desc}', game_desc)
    return prompt

def get_response(connector: LLMConnector, sample: dict, i: int, fs_subseed) -> dict:
    if not unified_fewshot:
        fewshot = get_single_sample_fewshot(samples, i, fs_subseed)
        # print(i, samples[i]['game_id'], fewshot[0]['game_id'], fewshot[1]['game_id'], fewshot[2]['game_id'], fewshot[3]['game_id'])
        initiate_conversation(connector, prompt, fewshot)
    request = sample_to_request(sample)
    response = ask_until_fail(json.dumps(request), connector)
    llm_sample = dict([(f'pred_{key}', value) for key, value in response.items()])
    return sample_to_response(sample)|llm_sample

def prepare_samples(samples: list[dict], max_count: int, sampling_rnd: random.Random, unifrom_game_id_sampling: bool) -> list[dict]:
    sampling_rnd.shuffle(samples)
    if unifrom_game_id_sampling:
        sample_by_game_id = {}
        for sample in samples:
            game_id = sample['game_id']
            sample_by_game_id[game_id] = sample_by_game_id.get(game_id, []) + [sample]
        game_ids = list(sample_by_game_id.keys())
        print(f"Total game count: {len(game_ids)}")
        default_count_per_game = max_count // len(game_ids)
        sample_count_per_game_id = {}
        for game_id in game_ids:
            sample_count_per_game_id[game_id] = min(default_count_per_game, len(sample_by_game_id[game_id]))
        if sum(sample_count_per_game_id.values()) != max_count:
            remaining = max_count - sum(sample_count_per_game_id.values())
            input(f"WARNING: THIS IS IMPORTANT! WE COULDNT SAMPLE UNIFORMLY, BECAUSE NOT ENOUGH SAMPLES PER GAME EXIST. {remaining} SAMPLES ARE ADDED RANDOMLY. PRESS ENTER TO CONTINUE")
            assert sum(sample_count_per_game_id.values()) <= max_count
            while sum(sample_count_per_game_id.values()) != max_count:
                order = [game_id for game_id in game_ids]
                sampling_rnd.shuffle(order)
                for game_id in order:
                    if sample_count_per_game_id[game_id] < len(sample_by_game_id[game_id]):
                        sample_count_per_game_id[game_id] += 1
                    if sum(sample_count_per_game_id.values()) == max_count:
                        break
        print(f"Sampling {min(sample_count_per_game_id.values())} to {max(sample_count_per_game_id.values())} samples per game")
        print("samples:", list(sample_count_per_game_id.values()))
        for val in range(max(sample_count_per_game_id.values()) - 1):
            existing = sum([1 if sample_count_per_game_id[game_id] == val else 0 for game_id in game_ids])
            count = sum([1 if len(sample_by_game_id[game_id]) == val else 0 for game_id in game_ids])
            assert existing == count, f"problem with value {val}: some games with this number of samples are not maxxed out. games:{count} != sampling:{existing}"
        input("Press enter to continue")
        new_samples = []
        for game_id in game_ids:
            new_samples += sample_by_game_id[game_id][:sample_count_per_game_id[game_id]]
        assert len(new_samples) == max_count
        return new_samples
    else:
        return samples[:max_count]

if __name__ == '__main__':
    with open(prompt_filename, 'r') as f:
        prompt_template = f.read()
    dataset_filename: str = sys.argv[1]
    max_count: int = int(sys.argv[2])
    seed: int = int(sys.argv[3])
    with open(dataset_filename, 'r') as f:
        dataset = json.load(f)
    game_name: str = dataset['name']
    game_desc: str = dataset['description']
    sampling_rnd = random.Random(seed)
    samples: list[dict] = prepare_samples(dataset['samples'], max_count, sampling_rnd, unifrom_game_id_sampling)
    fewshot: list[dict]
    prompt = fill_prompt(prompt_template, game_name, game_desc)
    connector = OpenAIChat(OpenAIChat.OpenAIModel.GPT_4O_mini)
    # connector = DeepSeekChat(DeepSeekChat.DeepSeekModel.DEEP_SEEK_CHAT)
    # connector = GeminiChat(GeminiChat.GeminiModel.Gemini_15_Flash_002)
    if unified_fewshot:
        samples, fewshot = fewshot_split(samples, fewshot_seed)
        initiate_conversation(connector, prompt, fewshot)
    fewshot_rnd = random.Random(fewshot_seed) # only used for non-unified
    result_samples: list[dict] = Parallel(n_jobs=thread_count)(delayed(get_response)(connector.copy(), sample, i, fewshot_rnd.randint(0, 1000000)) for i, sample in enumerate(tqdm(samples))) # type: ignore
    results = {
        'llm': connector.model_name,
        'dataset': dataset_filename,
        'initial_conversation': copy.deepcopy(connector.chat_log),
        'samples': result_samples
    }
    results['sample_count'] = len(result_samples)
    results['legal_accuracy'] = sum([1 if sample['legal'] == sample['pred_legal'] else 0 for sample in results['samples']])
    results['next_state_accuracy'] = sum([1 if sample['next_state'] == sample['pred_next_state'] else 0 for sample in results['samples']])
    results['legal_next_state_accuracy'] = sum([1 if sample['next_state'] == sample['pred_next_state'] else 0 for sample in results['samples'] if sample['legal']])
    results['legal_count'] = sum([1 for sample in results['samples'] if sample['legal']])

    base_dataset_name = os.path.splitext(os.path.basename(dataset_filename))[0]
    out_filename = os.path.join(f'results', f'{base_dataset_name}_{connector.model_name}_seed_{seed}_{unifrom_game_id_sampling}_{unified_fewshot}_{int(time.time())}.json')
    with open(out_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved as {out_filename}")
