import torch
import random
from typing import List
import re


def sokoban_projection(actions: List[str]):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <think>some reasoning...</think><action>up/down/left/right/still</action>
    Sokoban action mappings:
    - 0: No Op (Invalid Action)
    - 1: Push Up
    - 2: Push Down
    - 3: Push Left
    - 4: Push Right
    - 5: Up
    - 6: Down
    - 7: Left
    - 8: Right
    """

    action_pools = {
        "no_op": 0,
        "push up": 1,
        "push down": 2,
        "push left": 3,
        "push right": 4,
        "up": 5,
        "down": 6,
        "left": 7,
        "right": 8
    }

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = 0  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()

            for act in action_pools.keys():
                if act in extracted_action:
                    actions[i] = action_pools[act]
                    # if found legal action, set valids to 1
                    valids[i] = 1
                    break

            # If no valid action found, randomly select from pool
            if valids[i] == 0:
                actions[i] = 0

        except:
            # randomly choose an action from the action list if illegal
            actions[i] = 0

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

    return actions, valids