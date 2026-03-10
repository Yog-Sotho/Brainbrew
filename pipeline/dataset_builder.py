import json
from tqdm import tqdm

def build_with_vllm(prompts, engine, output_path, batch_size):

    with open(output_path,"w") as f:

        for i in tqdm(range(0,len(prompts),batch_size)):

            batch = prompts[i:i+batch_size]

            outputs = engine.generate_batch(batch)

            for p,o in zip(batch,outputs):

                rec = {
                    "instruction":p,
                    "output":o
                }

                f.write(json.dumps(rec)+"\n")
