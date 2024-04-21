import random
import numpy as np
import tqdm as tqdm

def get_init_prompt_chat(scenario_info_text, scenario_test_prompt, mc_gen_prompt_template):
    mc_gen_prompt_test = []
    num_data = len(scenario_info_text)

    for i in range(num_data):
        scenario_text = scenario_info_text[i]
        scene_descriptions = scenario_text.split("\n")[1].split("Scene: ")[1]
        instruction = scenario_text.split("\n")[2].split("Task: ")[1]
        new_prompt = scenario_test_prompt.format(scene_descriptions, instruction).strip()
        msg = mc_gen_prompt_template.strip() + "\n\n" + new_prompt
        mc_gen_prompt_test.append(msg)
        
    test_set = []
    for i in range(num_data):
        test_set.append({
            'info': scenario_info_text[i],
            'mc_gen_prompt': mc_gen_prompt_test[i],
        })
    return test_set

def get_reason_prompt(prompt):
    all_examples = prompt.split("\n\n")
    sys_prompt = all_examples[0].strip()
    messages=[{"role": "system", "content": sys_prompt}]
    for i in range(1, len(all_examples)-1):
        scenario = all_examples[i].split("You: ")
        example = scenario[0] + "You:"
        output = scenario[1].strip()
        new_msg_user = {"role": "system", "name": "example_user", "content": example}
        new_msg_assistant = {"role": "system", "name": "example_assistant", "content": output}
        messages.extend([new_msg_user, new_msg_assistant])
    
    scenario = all_examples[-1].strip()
    messages.append({"role": "user", "content": scenario})
    return messages

def get_pred_prompt(prompt):
    all_examples = prompt.split("\n\n")
    sys_prompt = all_examples[0].strip()
    messages=[{"role": "system", "content": sys_prompt}]
    for i in range(1, len(all_examples)-1):
        scenario = all_examples[i].split("Explain: ")
        example = scenario[0].strip() + "\nExplain:"
        output = scenario[1].strip()
        new_msg_user = {"role": "system", "name": "example_user", "content": example}
        new_msg_assistant = {"role": "system", "name": "example_assistant", "content": output}
        messages.extend([new_msg_user, new_msg_assistant])
    
    scenario = all_examples[-1].strip()
    messages.append({"role": "user", "content": scenario})
    return messages

def get_pred_prompt2(prompt):
    all_examples = prompt.split("\n\n")
    sys_prompt = all_examples[0].strip()
    messages=[{"role": "system", "content": sys_prompt}]
    for i in range(1, len(all_examples)-1):
        scenario = all_examples[i].split("Options:")
        example = scenario[0] + "Options:"
        output = scenario[1].strip()
        new_msg_user = {"role": "system", "name": "example_user", "content": example}
        new_msg_assistant = {"role": "system", "name": "example_assistant", "content": output}
        messages.extend([new_msg_user, new_msg_assistant])
    
    scenario = all_examples[-1].strip()
    messages.append({"role": "user", "content": scenario})
    return messages

def get_init_test_set_prompt_chat(scenario_info_text, scenario_test_prompt, prompt_template):
    all_examples = prompt_template.split("\n\n")
    sys_prompt = all_examples[0].strip()
    messages=[{"role": "system", "content": sys_prompt}]
    for i in range(1, len(all_examples)):
        scenario = all_examples[i].split("Options:")
        example = scenario[0] + "Options:"
        output = scenario[1].strip()
        new_msg_user = {"role": "system", "name": "example_user", "content": example}
        new_msg_assistant = {"role": "system", "name": "example_assistant", "content": output}
        messages.extend([new_msg_user, new_msg_assistant])
    
    mc_gen_prompt_test = []
    num_data = len(scenario_info_text)

    for i in range(num_data):
        scenario_text = scenario_info_text[i]
        scene_descriptions = scenario_text.split("\n")[1].split("Scene: ")[1]
        instruction = scenario_text.split("\n")[2].split("Task: ")[1]
        new_prompt = scenario_test_prompt.format(scene_descriptions, instruction).strip()
        msg = messages.copy()
        msg.append({"role": "user", "content": new_prompt})     
        mc_gen_prompt_test.append(msg)
        
    test_set = []
    for i in range(num_data):
        test_set.append({
            'info': scenario_info_text[i],
            'mc_gen_prompt': mc_gen_prompt_test[i],
        })
    return test_set

def get_init_scenario_prompt_chat(scenario_info_text, scenario_test_prompt):
    mc_gen_prompt_test = []
    num_data = len(scenario_info_text)

    for i in range(num_data):
        scenario_text = scenario_info_text[i]
        scene_descriptions = scenario_text.split("\n")[1].split("Scene: ")[1]
        instruction = scenario_text.split("\n")[2].split("Task: ")[1]
        new_prompt = scenario_test_prompt.format(scene_descriptions, instruction).strip()
        msg = [{"role": "user", "content": new_prompt}] 
        mc_gen_prompt_test.append(msg)
        
    test_set = []
    for i in range(num_data):
        test_set.append({
            'info': scenario_info_text[i],
            'mc_gen_prompt': mc_gen_prompt_test[i],
        })
    return test_set