class PromptTemplate:
    instruction = """You find yourself locked inside a mysterious room, and your ultimate goal is to escape the room.\n
You can explore the room, interact with objects, inspect items, and solve puzzles. 
Meanwhile, you should recover the entire story based on your findings, including useful items and obtained notes. 
You can adopt the following actions to explore the room and interact with objects:"""

    operations = """
- move_forward: float, ranged between [-10, 10]. This is the number of meters you want to move forward (negative value means moving backward).
- rotate_right: float, ranged between [-180, 180]. This is the number of degrees you want to turn right (negative value means turn left).
- rotate_down: float, ranged between [-90, 90]. This is the angle you want to adjust your view vertically. Positive value means looking downward, while a negative value means looking upward. Angle 0 means looking straight ahead. 
- jump: bool, whether you want to jump (can be used together with moving forward), e.g., True represents the action "to jump".
- look_at: list[x: foat, y: float], the range of x and y is [0, 1]. This parameter is the coordinates of the point in the image you want to look at. For reference, the coordinates of the upper left corner of the scene are (0, 0) and the coordinates of the lower right corner are (1, 1). Also to mention that there are on clues on the ceiling.
- grab: bool, whether you require to interact with the object located exactly at the center of the scene (marked by a red dot). e.g., to grab the key or to interact with (or open) a box at the center of the scene, set grab=True. The red dot assists in locating the object you require to interact with. You might need to adjust the view or move closer to ensure the red dot is on your target object, through the rotate_right, rotate_down, and move_forward actions. To successfully grab an object, you should center the object via the red dot and be in a certain distance to it. If the grabbing fails, try move closer towards the object. If it fails multiple times at the same position, you should be aware that not all objects are interactable, do not get stucked in uninteractable position.
- trigger: bool, whether you want to attempt to trigger the sound of a sound-emitting object (e.g., recorder/radiogram/musicbox) at the center of the scene. This ONLY works when you are close enough and facing it; otherwise it will fail and you should move closer / adjust your view.
- interactions : dict:{"use_item_id": str, this is the item_id you require to view or use (when used together with grab=True, it means to use this item to interact with the target object you want to grab, e.g. using item_id of the key to open the door in the scene), "input": str, this is the message you want to input when interacting with the center object}.
- read: str, this is the item_id that you want to get detailed information from your bag.
- rationale: str, represents the rationale of your action. This should explain your decision-making process and help the agent understand your thinking process.

You need to return data in the following format of JSON_string (in English) to interact with the scene(outputs must be in English) and don't say anything else:
{
    "move_forward": float,
    "rotate_right": float,
    "rotate_down": float,
    "jump": bool,
    "look_at": list[x: float, y: float],
    "grab": bool,
    "trigger": bool,
    "interactions": {
        "use_item_id": str,
        "input": str
    },
    "read": str,
    "rationale": str
}

All of the above operations are optional. If no value is passed in, the interactive operation will not be performed.

You must follow the above instructions and don't say anything else except for the JSON_string of operations.
"""
    SYS_PROMPT = instruction + operations

    STEP_PROMPT = """{interaction_result}
===
The items in your bag usable include:
{bag_desc}
===
Please determine the next action(s) that could help you observe the room or obtain useful tools or clues.
If you find yourself stuck in a corner, try turn around by passing rotate_right.
You need to return data in the following format of JSON_string (in English) to interact with the scene and don't say anything else: 
{{
    "move_forward": float,
    "rotate_right": float,
    "rotate_down": float,
    "jump": bool,
    "look_at": list[x: float, y: float],
    "grab": bool,
    "interactions": {{
        "use_item_id": str,
        "input": str
    }},
    "read": str,
    "rationale": str
}}
"""

    story_prompt = """You have successfully escaped the room. Now, reconstruct the entire story based on the items you discovered during the game and the overall environment you observed. Follow the steps below to guide your recollection and piece together the full narrative.

Step 1: Describe the room environment
"Begin by describing the room where you started. What did the room look like? What was the overall atmosphere? Were there any notable features, such as furniture, lighting, or strange objects? Include sensory details like smells, sounds, and the arrangement of the room. This will help set the scene for the story."

Step 2: Recall the items that may contain information or clues
"Think back to the objects you found throughout the game. What items did you come across? Were any of them unusual or seemed important? These could include physical items like keys, notes, or devices, or even abstract clues like symbols or markings on the wall. Reflect on how each item might have connected to the next step in your escape."

Step 3: Piece together the whole story
"Now, use the information from the room description and the items you've found to piece together the full story. What was the purpose of the room? Who or what might have created the escape challenge, and why? What was the sequence of events that led you to the escape? Try to connect the dots between the environment, the clues, and the items you encountered, and reconstruct the narrative from start to finish."
"""

    INTERACTION_SCHEMA = {
        "type": "object",
        "properties": {
            "move_forward": {"type": "number"},
            "rotate_right": {"type": "number", "minimum": -180, "maximum": 180},
            "rotate_down": {"type": "number", "minimum": -90, "maximum": 90},
            "jump": {"type": "boolean"},
            "look_at": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1},
                "minItems": 0,
                "maxItems": 2,
            },
            "grab": {"type": "boolean"},
            "trigger": {"type": "boolean"},
            "interactions": {
                "type": "object",
                "properties": {
                    "use_item_id": {"type": "string"},
                    "input": {"type": "string"},
                },
                "required": ["use_item_id", "input"],
            },
            "read": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "dependencies": {
            "interactions": {
                "properties": {"grab": {"const": True}},
                "required": ["grab"],
            }
        },
        "anyOf": [
            {"required": ["move_forward"]},
            {"required": ["move_right"]},
            {"required": ["rotate_right"]},
            {"required": ["rotate_down"]},
            {"required": ["jump"]},
            {"required": ["look_at"]},
            {"required": ["grab"]},
            {"required": ["trigger"]},
            {"required": ["interactions"]},
            {"required": ["read"]},
            {"required": ["rationale"]},
        ],
    }


class PromptTemplate_Base(PromptTemplate):
    instruction = """You find yourself locked inside a room, and your ultimate goal is to escape the room. i.e. the room escape game.\n
You can explore the room, interact with objects, inspect items, and resolve puzzles. You can adopt the following actions to explore the room and interact with objects:"""

    SYS_PROMPT = instruction + PromptTemplate.operations

    SYS_PROMPT_KEYONLY = (
        instruction
        + PromptTemplate.operations
        + "\nDuring the game, you will be provided with only key interaction steps containing useful information, some intemediate redundant steps will be omitted. "
    )


class PromptTemplate_Base2(PromptTemplate):
    instruction = """You find yourself locked inside a room, and your ultimate goal is to escape the room. i.e. the room escape game.\n
You can explore the room, interact with objects, inspect items, and resolve puzzles. If you find doors locked or uninteractable, you probably need to search for keys or passwords to unlock the door when interacting with the environment. You can adopt the following actions to explore the room and interact with objects:"""

    SYS_PROMPT = instruction + PromptTemplate.operations

    SYS_PROMPT_KEYONLY = (
        instruction
        + PromptTemplate.operations
        + "\nDuring the game, you will be provided with only key interaction steps containing useful information, some intemediate redundant steps will be omitted. "
    )


class PromptTemplate_Hint(PromptTemplate_Base):
    initial_hint = "===\nHere are some hints: you should get out through a door; if the door is locked, it can be unlocked by key or password, and you need to find the key or password to the door by exploring the room."
    SYS_PROMPT = (
        PromptTemplate_Base.instruction + PromptTemplate.operations + initial_hint
    )


