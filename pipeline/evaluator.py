import json

def evaluate_dataset(input_path, output_path, judge, threshold):

    with open(input_path) as fin, open(output_path,"w") as fout:

        for line in fin:

            obj=json.loads(line)

            score=judge.score(
                obj["instruction"],
                obj["output"]
            )

            if score >= threshold:

                obj["score"]=score
                fout.write(json.dumps(obj)+"\n")
