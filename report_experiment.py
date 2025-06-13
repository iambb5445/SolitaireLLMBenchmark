import sys
import json
from typing import Callable

def print_per_move(samples: list[dict], counts: Callable[[dict], bool], is_correct: Callable[[dict], bool]):
    move_type_count = dict()
    move_type_correct = dict()
    for sample in samples:
        if not counts(sample):
            continue
        move_type = sample['action'].split()[0]
        correct = is_correct(sample)
        move_type_count[move_type] = move_type_count.get(move_type, 0) + 1
        move_type_correct[move_type] = move_type_correct.get(move_type, 0) + (1 if correct else 0)
    for k in move_type_count.keys():
        print(f"{k}: {move_type_correct[k]}/{move_type_count[k]} - {move_type_correct[k]/move_type_count[k]}")

if __name__ == '__main__':
    results_filename = sys.argv[1]
    with open(results_filename, 'r') as f:
        results = json.load(f)
    if 'is_valid' in results['samples'][0]:
        # dataset report
        sample_count = len(results['samples'])
        legal_count = sum([1 for sample in results['samples'] if sample['is_valid']])
        print(f'sample count: {sample_count}')
        print(f'legal count: {legal_count}')
        print('Legal count per move')
        print_per_move(results['samples'], lambda _: True, lambda sample: sample['is_valid'])
        exit()
    sample_count = len(results['samples'])
    legal_count = sum([1 for sample in results['samples'] if sample['legal']])
    legal_correct = sum([1 if sample['legal'] == sample['pred_legal'] else 0 for sample in results['samples']])
    next_state_correct = sum([1 if sample['next_state'] == sample['pred_next_state'] else 0 for sample in results['samples']])
    legal_next_state_correct = sum([1 if sample['next_state'] == sample['pred_next_state'] else 0 for sample in results['samples'] if sample['legal']])
    print(f'sample count: {sample_count}')
    print(f'legal count: {legal_count}')
    print(f'legal accuracy: {legal_correct}/{sample_count} - {legal_correct / sample_count}')
    print(f'next state accuracy: {next_state_correct}/{sample_count} - {next_state_correct / sample_count}')
    print(f'legal next state accuracy: {legal_next_state_correct}/{legal_count} - {legal_next_state_correct / legal_count}')
    print('accuracy per action type [legal]:')
    print_per_move(results['samples'], lambda _: True, lambda sample: sample['legal'] == sample['pred_legal'])
    print('accuracy per action type [next_state]:')
    print_per_move(results['samples'], lambda sample: sample['legal'], lambda sample: sample['next_state'] == sample['pred_next_state'])