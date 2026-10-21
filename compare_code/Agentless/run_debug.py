import os
import sys
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7893"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7893"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
sys.path.insert(0, "/root/tangou1/LLM/LLMDD/compare_code/Agentless")
os.chdir("/root/tangou1/LLM/LLMDD/compare_code/Agentless")
from dotenv import load_dotenv

load_dotenv()
import runpy

# python agentless/fl/localize.py --file_level --output_folder ../../results/results_agentless/AgentIssue-Bench/AGiXT__AGiXT-1369/file_level --num_threads 1 --skip_existing --backend openai --model gemini-2.5-flash --dataset_json /root/tangou1/LLM/LLMDD/compare_code/Agentless/temp/tmpaqnv0of9.json

#   0%|          | 0/1 [00:00<?, ?it/s]
# 100%|██████████| 1/1 [00:00<00:00, 3039.35it/s]

# 步骤 2/ 8...
# python agentless/fl/localize.py --file_level --irrelevant --output_folder ../../results/results_agentless/AgentIssue-Bench/AGiXT__AGiXT-1369/file_level_irrelevant --num_threads 1 --skip_existing --backend openai --model gemini-2.5-flash --dataset_json /root/tangou1/LLM/LLMDD/compare_code/Agentless/temp/tmpaqnv0of9.json

args = "python agentless/fl/localize.py --file_level --output_folder ../../results/results_agentless/AgentIssue-Bench/AGiXT__AGiXT-1369/file_level --num_threads 1 --skip_existing --backend openai --model gemini-2.5-flash --dataset_json /root/tangou1/LLM/LLMDD/compare_code/Agentless/temp/tmpaqnv0of9.json"
args = args.split()

if args[0] == "python":
    args.pop(0)

if args and args[0] == "-m":
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path

if args:
    sys.argv.extend(args[1:])
    fun(args[0], run_name="__main__")
