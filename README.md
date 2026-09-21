# Benchmark Testing

# _____________________
**Username: benchmark**             
**Password: benchmark**
# _____________________

**The SSD should already have everything downloaded and ready to run**

**If the model is not working or downloaded correctly causing errors follow this link**
**https://huggingface.co/openai/gpt-oss-20b**

**Use vLLM Model**

# Make the virtual environment to run everything
**Open the terminal and put this in**

```instruction
source .venv/bin/activate
```


# Now in the terminal you should see       
**(benchmark) benchmark@benchmark-AS-4125GS-TNRT2** 
# Start with next step first (Only use this if model doesn't run)
**If model is not loaded when running oss-20b**
```Instruction
uv pip install --pre vllm==0.10.1+gptoss 

    --extra-index-url https://wheels.vllm.ai/gpt-oss/ 

    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 

    --index-strategy unsafe-best-match 

```
# Next steps are to be done together
**This will run the oss-20b**
```instruction
vllm serve openai/gpt-oss-20b --tensor-parallel-size 4
```
# In a different Terminal
**This will Collect data and save it too a file**
```instruction
while true; do 

    nvidia-smi --query-gpu=timestamp,index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader >> gpu_metrics.csv 

    sleep 2 

done
```

# In a third terminal (Optional)
**This will open a live window showing the data**
```Instruction
watch -n 2 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv'
```
 
