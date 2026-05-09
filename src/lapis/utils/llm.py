from copy import deepcopy
import os

def export_result(response: str, file_name: str):
    if "```" in response:
        response = response.split("```")[1]
    start_idx = response.find("(define")
    if start_idx != 0 and response[0: start_idx] != "\n":
        response = response.replace(response[0: start_idx], "")
    
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    with open(file_name, "w") as f:
        f.write(response)

def export_subgoal_list(response: str):
    subgoal_list = []
    for g in response.split("```"):
        cg = deepcopy(g)
        cg = cg.replace(" ", "")
        if "(:goal" in cg and cg.endswith(")\n"):
            subgoal_list.append(g[g.find("(:goal"):g.rfind(")\n")+len(")\n")])
    return subgoal_list