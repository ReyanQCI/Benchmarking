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
#!/bin/bash

echo "timestamp,cpu_temp_c,gpu_index,gpu_temp_c,gpu_utilization_pct,gpu_power_w,gpu_memory_used_mib" > Datalog.csv

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Get CPU temperature (adjust this if your sensors output has multiple temp sensors)
    cpu_temp=$(sensors | awk '/^Tctl:/ {gsub(/[+°C]/,"",$2); print $2; exit}')

    nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,power.draw,memory.used \
        --format=csv,noheader,nounits | \
    while IFS=',' read -r gpu_index gpu_temp gpu_util gpu_power gpu_memory; do
        gpu_index=$(echo "$gpu_index" | xargs)
        gpu_temp=$(echo "$gpu_temp" | xargs)
        gpu_util=$(echo "$gpu_util" | xargs)
        gpu_power=$(echo "$gpu_power" | xargs)
        gpu_memory=$(echo "$gpu_memory" | xargs)

        echo "$timestamp,$cpu_temp,$gpu_index,$gpu_temp,$gpu_util,$gpu_power,$gpu_memory" >> Datalog.csv
    done

    sleep 2
done

```


# In a third terminal Run this file to constantly send requests
**Already set up as request.py**
```Instruction
import asyncio
import aiohttp
import os
import sys
from datetime import datetime


URL = "http://localhost:8000/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

NUM_WORKERS = 32

#==========
# For if it takes too long to complete request ie(Error) it will stop running
#==========

REQUEST_TIMEOUT = aiohttp.ClientTimeout(
    total=60,
    connect=10,
    sock_read=50,
)

#==============
# Sends a request to the model to complete
# At 512 Tokens the GPUs are are 100% use
#==============

async def request(session, worker_id):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Explain the theory of relativity in substantial detail.",
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }

    async with session.post(URL, json=payload) as response:
        response.raise_for_status()

        body = await response.read()

        if not body:
            raise RuntimeError("Server returned an empty response")


async def worker(session, worker_id):
    while True:
        await request(session, worker_id)


async def main():
    print(f"Workers: {NUM_WORKERS}")  #Workers are loaded requests (For this 32 requests are sent)
    print(f"Model:   {MODEL}")
    print(f"URL:     {URL}")
    print("Running Benchmark...")

    #==========
    # Showing an output to see if it is running
    #==========

    connector = aiohttp.TCPConnector(
        limit=NUM_WORKERS,
        limit_per_host=NUM_WORKERS,
    )

    async with aiohttp.ClientSession(
        timeout=REQUEST_TIMEOUT,
        connector=connector,
    ) as session:

        #=========
        # All workers keep creating tasks
        #=========

        workers = [
            asyncio.create_task(worker(session, i + 1))
            for i in range(NUM_WORKERS)
        ]

        try:
            await asyncio.gather(*workers)

        #============
        # Make sure no errors our found or else it will stop sending requests
        #============

        except Exception as error:

            log_file = "Error.txt"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(log_file, "a", encoding = "utf-8") as file:
                file.write("=" * 60 + "\n")
                file.write("TEST FAILED \n")
                file.write(f"Time:  {timestamp} \n")
                file.write(f"Error: {type(error).__name__}: {error} \n")
                file.write("=" * 60 + "\n")

            #========
            # Stop all remaining workers.
            #========

            for task in workers:
                task.cancel()

            #========
            #Stops all 32 workers
            #========

            await asyncio.gather(
                *workers,
                return_exceptions=True,
            )
            os.system("shutdown -h +1")
            sys.exit(1)

#=========
# Run until Ctrl+C is pressed to stop it
#=========

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest manually stopped.")
        sys.exit(130)

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
