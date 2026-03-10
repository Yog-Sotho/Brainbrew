import json

def export_alpaca(input_path, output_path):

    with open(input_path) as fin, open(output_path,"w") as fout:

        for line in fin:

            obj=json.loads(line)

            rec={
                "instruction":obj["instruction"],
                "input":"",
                "output":obj["output"]
            }

            fout.write(json.dumps(rec)+"\n")
