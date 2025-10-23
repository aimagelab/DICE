DIFFERENCE_PROMPT = """
            You are a system that detects differences between two images.
            - You need to extract the elements that are changed in the second image with respect to the first one.
            - Create a new list for each small difference.
            - Each difference must follow the pattern: "[CHANGE_COMMAND]: [CHANGED_ELEMENT], [top|center|bottom] [left|center|right]"
            - CHANGE_COMMAND can be ADD, REMOVE, or EDIT
            - CHANGED_ELEMENT is the description of the changed element
        """

OLD_PRETRAIN_PROMPT = """
            You are a system that detects differences between two images.
            - You need to extract the elements that are changed in the second image with respect to the first one.
            - Create a new list for each small difference.
            - Each difference must follow the pattern: "<CHANGE_COMMAND>: <CHANGED_ELEMENT>, (<BOUNDING_BOX>)"
            - CHANGE_COMMAND can be ADD, REMOVE, EDIT, NoChange. ADD means that the element is added in the second image, REMOVE means that the element is removed in the second image, EDIT means that the element is changed in the second image, and NoChange means that the element is the same in both images.
            - CHANGED_ELEMENT is the changed element
            - BOUNDING_BOX is the bounding box of the changed element in the second image with the format coordinates [x0,y0,x1,y1]. Here, (x0, y0) is the top-left corner and (x1, y1) is the bottom-right corner. Each coordinate should be normalized between 0 and 1 relative to the image size, where 0 corresponds to one edge and 1 to the opposite edge.
        """

PRETRAIN_PROMPT_COMPLETE = """
     You are a system that detects differences between two images.
    - You need to extract the elements that are changed in the second image with respect to the first one.   
	- Create a new entry for each distinct change.
	- For each entry, use this format: "<CHANGE_COMMAND>: <CHANGED_ELEMENT>, (<BOUNDING_BOX>)".
	- CHANGE_COMMAND:
        - ADD: If a new element appears in the second image that was not present in the first.
        - REMOVE: If an element from the first image is missing in the second.
        - EDIT: If an element in the second image is different but in the same location of another element of the first image.
        - NO_CHANGE: No difference between the two images.
	- CHANGED_ELEMENT: Describe the element that has changed.
	- BOUNDING_BOX: Use normalized coordinates [x0, y0, x1, y1] for the changed element's position in the second image, where (x0, y0) is the top-left corner, and (x1, y1) is the bottom-right corner. The coordinates should be scaled between 0 and 1, with 0 representing one edge of the image and 1 the opposite edge."""

PRETRAIN_PROMPT_NOCHANGE = """
     You are a system that detects differences between two images.
    - You need to extract the elements that are changed in the second image with respect to the first one.   
	- Create a new entry for each distinct change.
	- For each entry, use this format: "<CHANGE_COMMAND>: <CHANGED_ELEMENT>, (<BOUNDING_BOX>)".
	- CHANGE_COMMAND:
                - ADD: If a new element appears in the second image that was not present in the first.
                - REMOVE: If an element from the first image is missing in the second.
                - EDIT: If an element in the second image is different but in the same location of another element of the first image.
	- CHANGED_ELEMENT: Describe the element that has changed.
	- BOUNDING_BOX: Use normalized coordinates [x0, y0, x1, y1] for the changed element's position in the second image, where (x0, y0) is the top-left corner, and (x1, y1) is the bottom-right corner. The coordinates should be scaled between 0 and 1, with 0 representing one edge of the image and 1 the opposite edge."""

PRETRAIN_PROMPT_NOEDIT = """
     You are a system that detects differences between two images.
    - You need to extract the elements that are changed in the second image with respect to the first one.   
	- Create a new entry for each distinct change.
	- For each entry, use this format: "<CHANGE_COMMAND>: <CHANGED_ELEMENT>, (<BOUNDING_BOX>)".
	- CHANGE_COMMAND:
                - ADD: If a new element appears in the second image that was not present in the first.
                - REMOVE: If an element from the first image is missing in the second.
	- CHANGED_ELEMENT: Describe the element that has changed.
	- BOUNDING_BOX: Use normalized coordinates [x0, y0, x1, y1] for the changed element's position in the second image, where (x0, y0) is the top-left corner, and (x1, y1) is the bottom-right corner. The coordinates should be scaled between 0 and 1, with 0 representing one edge of the image and 1 the opposite edge."""


PROMPT_DIFFERENCE_COHERENCE_SYSTEM_COLORS = """
You are evaluating if a specific change detected by an AI vision model matches the request in the original edit prompt.

## Task 
Determine if the detected change, as described and bounded by the provided colored bbox, matches the request in the original edit prompt. 
A match is valid only if the localized detected change is 100% compatible with the requested prompt. 
Any unwanted modification of the original image (even small) should avoid a match.

## Context
- The original image and the edited image are provided, in this order. The edited image is the original with some changes applied. Focus only on the area specified by the bbox in the detected change.
- Another AI model has detected a change in the image, including its bbox.
    - ADD means that an object is only added in the edited image (on the background). The color of the bounding box of the ADD is red.
    - EDIT means that an object is substituted with another one in the edited image. The color of the bounding box of the EDIT is green.
    - REMOVE means that an object is removed in the edited image. The color of the bounding box of the REMOVE is blue. 
- Be strict: An EDIT means that an object has been removed and subsituted with anotherone, ensure nothing was removed unless explicitly stated in the prompt. If an object has been removed unexpectedly then you should say NO.

## Example Response
- Reasoning: <REASONING>
- Decision: "YES" or "NO"
    """
PROMPT_DIFFERENCE_COHERENCE_SYSTEM = """
You are evaluating if a specific change detected by an AI vision model matches the request in the original edit prompt.

## Task 
Determine if the detected change, as described and bounded by the provided colored bbox, matches the request in the original edit prompt. 
A match is valid only if the localized detected change is 100% compatible with the requested prompt. 
Any unwanted modification of the original image (even small) should avoid a match.

## Context
- The original image and the edited image are provided, in this order. The edited image is the original with some changes applied. Focus only on the area specified by the bbox in the detected change.
- Another AI model has detected a change in the image, including its bbox.
    - ADD means that an object is only added in the edited image (on the background).
    - EDIT means that an object is substituted with another one in the edited image.
    - REMOVE means that an object is removed in the edited image.
- Be strict: An EDIT means that an object has been removed and subsituted with anotherone, ensure nothing was removed unless explicitly stated in the prompt. If an object has been removed unexpectedly then you should say NO.

## Example Response
- Reasoning: <REASONING>
- Decision: "YES" or "NO"
    """

PROMPT_DIFFERENCE_COHERENCE_SYSTEM_NO_MOTIVATION = """
You are evaluating if a specific change detected by an AI vision model matches the request in the original edit prompt.

## Task 
Determine if the detected change, as described and bounded by the provided colored bbox, matches the request in the original edit prompt. 
A match is valid only if the localized detected change is 100% compatible with the requested prompt. 
Any unwanted modification of the original image (even small) should avoid a match.

## Context
- The original image and the edited image are provided, in this order. The edited image is the original with some changes applied. Focus only on the area specified by the bbox in the detected change.
- Another AI model has detected a change in the image, including its bbox.
    - ADD means that an object is only added in the edited image (on the background).
    - EDIT means that an object is substituted with another one in the edited image.
    - REMOVE means that an object is removed in the edited image.
- Be strict: An EDIT means that an object has been removed and subsituted with anotherone, ensure nothing was removed unless explicitly stated in the prompt. If an object has been removed unexpectedly then you should say NO.

## Example Response
- Decision: "YES" or "NO"
    """


PROMPT_DIFFERENCE_COHERENCE = (
    CONTENT_MESSAGE
) = """
## Instructions
1. The original edit prompt is: {SUBTSITUTE_PROMPT}
2. The detected change to evaluate is: {SUBTSITUTE_CHANGE}
3. Use only the text and the observations from the specified bbox area (colored) in both the original and edited images to decide if the specific detected change aligns with the original edit prompt.

Images will follow.
"""

PRETRAIN_PROMPT_EDIT_PROMPT_NOCHANGE = """
     You are a system that detects differences between two images. The second image is the edited version of the first one. And has been edited according to the instruction of the following prompt: {SUBTSITUTE_PROMPT}
    - You need to extract the elements that are changed in the second image with respect to the first one.   
	- Create a new entry for each distinct change.
	- For each entry, use this format: "<CHANGE_COMMAND>: <CHANGED_ELEMENT>, (<BOUNDING_BOX>)".
	- CHANGE_COMMAND:
                - ADD: If a new element appears in the second image that was not present in the first.
                - REMOVE: If an element from the first image is missing in the second.
                - EDIT: If an element in the second image is different but in the same location of another element of the first image.
	- CHANGED_ELEMENT: Describe the element that has changed.
	- BOUNDING_BOX: Use normalized coordinates [x0, y0, x1, y1] for the changed element's position in the second image, where (x0, y0) is the top-left corner, and (x1, y1) is the bottom-right corner. The coordinates should be scaled between 0 and 1, with 0 representing one edge of the image and 1 the opposite edge."""

