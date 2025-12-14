import json
import os

def load_execution_data(exec_file_path):
    """Loads execution data and creates a map from question_id to times."""
    print(f"Loading execution data from {exec_file_path}...")
    with open(exec_file_path, 'r') as f:
        data = json.load(f)
    
    exec_map = {}
    for item in data:
        q_id = item.get('question_id')
        metadata = item.get('metadata')
        
        if q_id and metadata:
            # metadata should be a list, but handle dict just in case
            if isinstance(metadata, dict):
                metadata = [metadata]
            
            times_list = []
            for meta_item in metadata:
                if isinstance(meta_item, dict):
                    if 'times' in meta_item:
                        # Clean keys in times: remove 'candidate.TestCases.' prefix
                        times = meta_item['times']
                        cleaned_times = {}
                        for k, v in times.items():
                            clean_k = k.replace('candidate.TestCases.', '')
                            cleaned_times[clean_k] = v
                        times_list.append(cleaned_times)
                    elif 'error' in meta_item:
                        # Store error information if present
                        times_list.append({'_error': meta_item['error']})
                    else:
                        times_list.append(None)
                else:
                    times_list.append(None) # Keep None for indices with no times or error
            
            exec_map[q_id] = times_list
    
    print(f"Loaded execution times for {len(exec_map)} tasks.")
    return exec_map

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pred_file_path = os.path.join(base_dir, 'results/qwen3-coder-30B-A3B-instruct/test.jsonl')
    exec_file_path = os.path.join(base_dir, '../actual_exec/results/qwen3-coder-30B-A3B-instruct/nucleus_eval_all.json')
    # Output to the same directory as prediction file
    output_file_path = os.path.join(os.path.dirname(pred_file_path), 'time_ratios.jsonl')
    
    # exec_file_path might need adjustment if run from specific CWD, but using abspath from file location is safer
    # Check if exec file exists via relative path assumption, if not try absolute path from known structure
    if not os.path.exists(exec_file_path):
        # Fallback to absolute path based on user workspace structure if relative fail
        exec_file_path = '/Users/woo/Documents/Simtest_evaluation/BigCodeBench_Hard/actual_exec/results/qwen3-coder-30B-A3B-instruct/nucleus_eval_all.json'
        
    exec_map = load_execution_data(exec_file_path)
    
    print(f"Processing prediction data from {pred_file_path}...")
    
    results = []
    total_ratios = []
    
    with open(pred_file_path, 'r') as f_in, open(output_file_path, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            
            pred_item = json.loads(line)
            q_id = pred_item.get('id')
            
            if q_id not in exec_map:
                print(f"Warning: Execution data not found for {q_id}")
                continue
            
            code_index = pred_item.get('code_index', 0)
            exec_data_list = exec_map[q_id]
            
            if code_index >= len(exec_data_list):
                 print(f"Warning: Code index {code_index} out of bounds for {q_id} (len {len(exec_data_list)})")
                 continue
                 
            exec_times = exec_data_list[code_index]
            
            if not exec_times:
                 # No times (likely execution error)
                 continue
            
            # Check for error
            is_timeout = False
            if '_error' in exec_times and exec_times['_error'] == 'GlobalTimeout':
                is_timeout = True

            latency_list = pred_item.get('latency_list', {})
            
            task_ratios = {}
            total_task_predict = 0.0
            total_task_execute = 0.0
            
            for test_case, t_predict in latency_list.items():
                if is_timeout:
                    t_execute = 10.0 # GlobalTimeout assumption
                elif test_case in exec_times:
                    t_execute = exec_times[test_case]
                else:
                    # Test case not found in execution results (and not a global timeout)
                    continue
                    
                # Accumulate totals for task-level ratio (only for matching cases)
                total_task_predict += t_predict
                total_task_execute += t_execute

                if t_execute > 0:
                    ratio = t_predict / t_execute
                    task_ratios[test_case] = ratio
                    total_ratios.append(ratio)
                else:
                    # Handle zero execution time
                    if t_predict > 0:
                        ratio = float('inf') 
                    else:
                        ratio = 1.0 
                    task_ratios[test_case] = ratio 
            
            # Calculate per-task ratio
            if total_task_execute > 0:
                task_ratio = total_task_predict / total_task_execute
            elif total_task_predict > 0:
                task_ratio = float('inf')
            else:
                task_ratio = 1.0 if total_task_predict == 0 else 0.0 # 0/0 -> 1.0? or 0? Defaulting to 1.0 if both 0.

            result_entry = {
                "id": q_id,
                "task_ratio": task_ratio,
                "ratios": task_ratios
            }
            results.append(result_entry)
            f_out.write(json.dumps(result_entry) + '\n')
            
    if total_ratios:
        import numpy as np
        avg_ratio = np.mean([r for r in total_ratios if r != float('inf')])
        median_ratio = np.median([r for r in total_ratios if r != float('inf')])
        print(f"Processed {len(results)} tasks.")
        print(f"Average Latency Ratio: {avg_ratio:.2f}")
        print(f"Median Latency Ratio: {median_ratio:.2f}")
    else:
        print("No ratios calculated.")

if __name__ == "__main__":
    main()
