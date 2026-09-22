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

# Change nvcc
```Instruction
export CUDA_HOME=/usr/local/cuda
```
```Instruction
export PATH=/usr/local/cuda/bin:$PATH 
```
```Instruction
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} 
```
**Check for Cuda_13.4**
```Instruction
nvcc --version
```
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

    echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >> sensors.log 

    sensors >> sensors.log 

    sleep 2 

done 
```


# In a third terminal Run this file to constantly send requests
**Already set up as request.py**
```Instruction
import asyncio
import aiohttp

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

async def request(session):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Explain the theory of relativity in substantial detail."
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }

    async with session.post(URL, json=payload) as r:
        await r.read()

async def worker(session):
    while True:
        try:
            await request(session)
        except Exception as e:
            print(f"Request failed: {e}")
            await asyncio.sleep(1)

async def main():
    async with aiohttp.ClientSession() as session:
        workers = [
            asyncio.create_task(worker(session))
            for _ in range(32)
        ]
        await asyncio.gather(*workers)

asyncio.run(main())

```
**To run this**
```instruction
python3 request.py
```
# (Optional) In another Terminal
**This will open a live window showing the data**
```Instruction
watch -n 1 'echo "=== CPU ==="; sensors; echo; echo "=== GPUs ==="; nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,power.draw,memory.used --format=csv'
```

# To end the tasks press ctrl+C 
