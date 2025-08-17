#! /bin/bash
# python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/open_assistant/open_assistant_results_oai_format.jsonl --output_dir results/open_assistant --min_cluster_size 30 --run_metrics
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/self_instruct/self_instruct_results_oai_format.jsonl --output_dir ../Whatever-this-is/self_instruct_stratified --min_cluster_size 30 --groupby_column behavior_type
# python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/vicuna/vicuna_results_oai_format.jsonl --output_dir results/vicuna --min_cluster_size 30 --run_metrics
# python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/koala/koala_results_oai_format.jsonl --output_dir results/koala --min_cluster_size 30 --run_metrics
# python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/helm/grammar_results_oai_format.jsonl --output_dir results/grammar --min_cluster_size 30 --run_metrics
# python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/helm/helm_wildbench_results_oai_format.jsonl --output_dir results/wildbench --min_cluster_size 30 --run_metrics

python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/medhelm/aci_bench_results_oai_format.jsonl --output_dir ../Whatever-this-is/aci_bench_stratified --min_cluster_size 30 --groupby_column behavior_type
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/helm/helm_bigcodebench_results_processed_oai_biggest_models.jsonl --output_dir ../Whatever-this-is/bigcodebench_biggest_models_stratified --groupby_column behavior_type

python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/taubench/retail_data_oai_format.jsonl --output_dir ../Whatever-this-is/taubench_retail_stratified --system_prompt agent_system_prompt --groupby_column behavior_type
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/taubench/retail_data_oai_format.jsonl --output_dir ../Whatever-this-is/taubench_retail --system_prompt agent_system_prompt
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/taubench/airline_data_oai_format.jsonl --output_dir ../Whatever-this-is/taubench_airline_stratified --system_prompt agent_system_prompt --groupby_column behavior_type

# python scripts/run_pipeline.py --method side_by_side --use_wandb --input_file data/arena_webdev_sbs.jsonl --output_dir results/arena_webdev_sbs --system_prompt webdev_system_prompt_no_examples --min_cluster_size 15 --run_metrics
# python scripts/run_pipeline.py --method side_by_side --use_wandb --input_file data/arena_sbs.jsonl --output_dir results/arena_sbs --system_prompt sbs_w_metrics_system_prompt --min_cluster_size 15 --run_metrics