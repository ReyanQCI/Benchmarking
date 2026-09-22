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

    echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >> sensors.csv 

    sensors >> sensors.csv

    sleep 2 

done 
```


# In a third terminal Run this file to constantly send requests
**Already set up as request.py**
```Instruction
import asyncio
import aiohttp
import sys
from datetime import datetime


URL = "http://localhost:8000/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

NUM_WORKERS = 32

REQUEST_TIMEOUT = aiohttp.ClientTimeout( # For if it takes too long to complete request ie(Error) it will stop runnin
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
                "content": "Explain the theory of relativity in substantial detail."
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
    print(f"Workers: {NUM_WORKERS}") # Workers are loaded requests (For this 32 requests are sent)
    print(f"Model:   {MODEL}")
    print(f"URL:     {URL}")
    print(f"Running Benchmark...")
    # Showing an output to see if it is running


    connector = aiohttp.TCPConnector(
        limit=NUM_WORKERS,
        limit_per_host=NUM_WORKERS,
    )

    async with aiohttp.ClientSession(
        timeout=REQUEST_TIMEOUT,
        connector=connector,
    ) as session:

        workers = [ # All workers keep creating tasks
            asyncio.create_task(worker(session, i + 1))
            for i in range(NUM_WORKERS)
        ]

        try:
            await asyncio.gather(*workers)

        except Exception as error: # Make sure no errors our found or else it will stop sending requests
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print()
            print("=" * 60)
            print("TEST FAILED")
            print(f"Time:  {timestamp}")
            print(f"Error: {type(error).__name__}: {error}")
            print("=" * 60)

            # Stop all remaining workers.
            for task in workers:
                task.cancel()

            await asyncio.gather( #Stops all 32 workers
                *workers,
                return_exceptions=True,
            )

            sys.exit(1)


if __name__ == "__main__": # Run until Ctrl+C is pressed to stop it
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
