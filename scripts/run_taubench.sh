#! /bin/bash
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/helm/helm_bigcodebench_results_processed.jsonl --output_dir results/bigcodebench --min_cluster_size 15 

python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/taubench/retail_data.jsonl --output_dir results/taubench_retail --system_prompt taubench_system_prompt 
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/taubench/retail_data_incorrect.jsonl --output_dir results/taubench_retail_incorrect_only --system_prompt taubench_system_prompt 
# python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/taubench/airline_data.jsonl --output_dir results/taubench_airline --system_prompt taubench_system_prompt 
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/taubench/airline_data_incorrect.jsonl --output_dir results/taubench_airline_incorrect_only --system_prompt taubench_system_prompt 

python scripts/run_pipeline.py --method side_by_side --use_wandb --input_file data/arena_webdev_sbs.jsonl --output_dir results/arena_webdev_sbs --system_prompt webdev_system_prompt_no_examples --min_cluster_size 15 
python scripts/run_pipeline.py --method side_by_side --use_wandb --input_file data/arena_sbs.jsonl --output_dir results/arena_sbs --system_prompt sbs_w_metrics_system_prompt --min_cluster_size 15 