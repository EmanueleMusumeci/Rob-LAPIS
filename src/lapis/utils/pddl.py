def get_pddl_domain_actions(domain: str):
    start = domain.find("; Begin actions\n") + len("; Begin actions\n")
    end = domain.find("\n    ; End actions", start)
    actions = domain[start:end]
    return actions

def get_pddl_domain_types(domain: str):
    start = domain.find("; Begin types\n") + len("; Begin types\n")
    end = domain.find("\n    ; End types", start)
    types = domain[start:end]
    return types

def get_pddl_domain_predicates(domain: str):
    start = domain.find("; Begin predicates\n") + len("; Begin predicates\n")
    end = domain.find("\n    ; End predicates", start)
    predicates = domain[start:end]
    return predicates

def export_subgoal_list(response: str):
    subgoal_list = []
    for g in response.split("```"):
        cg = deepcopy(g)
        cg = cg.replace(" ", "")
        if "(:goal" in cg and cg.endswith(")\n"):
            subgoal_list.append(g[g.find("(:goal"):g.rfind(")\n")+len(")\n")])
    return subgoal_list

def export_result(response: str, file_name: str):
    if "```" in response:
        response = response.split("```")[1]
    start_idx = response.find("(define")
    if start_idx != 0 and response[0: start_idx] != "\n":
        response = response.replace(response[0: start_idx], "")
    with open(file_name, "w") as f:
        f.write(response)


def set_pddl_problem_goal(p_path: str, new_goal: str):
    with open(p_path, 'r') as pf:
        content = pf.readlines()

    start_idx = content.index('    ; Begin goal\n') + 1
    end_idx = content.index('    ; End goal\n')
    content[start_idx:end_idx] = [new_goal]

    with open(p_path, 'w') as pf:
        pf.writelines(content)

def set_pddl_problem_init(p_path: str, new_init: list):
    new_init_str = "\t\t" + "\n\t\t".join(new_init) + "\n"

    with open(p_path, 'r') as pf:
        content = pf.readlines()

    start_idx = content.index('    (:init\n') + 1
    end_idx = content.index('    ; End init\n') - 1
    content[start_idx:end_idx] = [new_init_str]

    with open(p_path, 'w') as pf:
        pf.writelines(content)