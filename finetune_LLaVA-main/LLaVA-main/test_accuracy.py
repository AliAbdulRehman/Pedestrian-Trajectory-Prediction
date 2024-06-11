# import torch

# from llava.constants import (
#     IMAGE_TOKEN_INDEX,
#     DEFAULT_IMAGE_TOKEN,
#     DEFAULT_IM_START_TOKEN,
#     DEFAULT_IM_END_TOKEN,
#     IMAGE_PLACEHOLDER,
# )
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import (
#     process_images,
#     tokenizer_image_token,
#     get_model_name_from_path,
# )

# from PIL import Image

# import requests
# from PIL import Image
# from io import BytesIO
# import re
# import json


# def image_parser(image_file):
#     out = image_file.split(SEP)
#     return out


# def load_image(image_file):
#     if image_file.startswith("http") or image_file.startswith("https"):
#         response = requests.get(image_file)
#         image = Image.open(BytesIO(response.content)).convert("RGB")
#     else:
#         image = Image.open(image_file).convert("RGB")
#     return image


# def load_images(image_files):
#     out = []
#     for image_file in image_files:
#         image = load_image(image_file)
#         out.append(image)
#     return out


# def eval_model(pedestrian_status):
#     # Model
#     disable_torch_init()
#     print("Entered eval")

#     model_name = get_model_name_from_path(MODEL_PATH)
#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         MODEL_PATH, MODEL_BASE, model_name
#     )

#     qs = QUERY
#     image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
#     if IMAGE_PLACEHOLDER in qs:
#         if model.config.mm_use_im_start_end:
#             qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
#         else:
#             qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
#     else:
#         if model.config.mm_use_im_start_end:
#             qs = image_token_se + "\n" + qs
#         else:
#             qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

#     if "llama-2" in model_name.lower():
#         conv_mode = "llava_llama_2"
#     elif "mistral" in model_name.lower():
#         conv_mode = "mistral_instruct"
#     elif "v1.6-34b" in model_name.lower():
#         conv_mode = "chatml_direct"
#     elif "v1" in model_name.lower():
#         conv_mode = "llava_v1"
#     elif "mpt" in model_name.lower():
#         conv_mode = "mpt"
#     else:
#         conv_mode = "llava_v0"

#     #if CONV_MODE is not None and conv_mode != CONV_MODE:
#     #    print(
#     #        "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
#     #            conv_mode, CONV_MODE, CONV_MODE
#     #        )
#     #    )
#     #else:
#     CONV_MODE = "ped1"

#     conv = conv_templates[CONV_MODE].copy()
#     conv.append_message(conv.roles[0], qs)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()

#     image_files = image_parser(IMAGE_FILE)
#     images = load_images(image_files)
#     image_sizes = [x.size for x in images]
#     images_tensor = process_images(
#         images,
#         image_processor,
#         model.config
#     ).to(model.device, dtype=torch.float16)

#     input_ids = (
#         tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#         .unsqueeze(0)
#         .cuda()
#     )

#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids,
#             images=images_tensor,
#             image_sizes=image_sizes,
#             do_sample=True if TEMPERATURE > 0 else False,
#             temperature=TEMPERATURE,
#             top_p=TOP_P,
#             num_beams=NUM_BEAMS,
#             max_new_tokens=MAX_NEW_TOKENS,
#             use_cache=True,
#         )

#     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

#     if pedestrian_status in outputs:
#         present = 1
#     else:
#         present = 0
#     # print(outputs)
#     return present


# if __name__ == "__main__":
#     # Define arguments directly within the code
#     MODEL_PATH = "checkpoints0/llava-v1.5-7b-task-lora"
#     MODEL_BASE = "liuhaotian/llava-v1.5-7b"
#     IMAGE_FILE = "your_image.jpg"
#     QUERY = "Your query text here"
#     CONV_MODE = "ped1"
#     SEP = ","
#     TEMPERATURE = 0.2
#     TOP_P = None
#     NUM_BEAMS = 1
#     MAX_NEW_TOKENS = 512

#     with open('./playground/data/overall_val_prompts_fm_rev.json', 'r') as file:
#         data = json.load(file)

#     success = 0
#     total = 0

#     for entry in data:
#         total = total + 1
#         image_file = entry['image']
        
#         human_prompts = []
#         gpt_answers = []
#         pedestrian_statuses = []
        
#         conversations = entry['conversations']
        
#         for conversation in conversations:
#             if conversation['from'] == 'human':
#                 human_prompts.append(conversation['value'])
#             elif conversation['from'] == 'gpt':
#                 gpt_answer = conversation['value']
#                 gpt_answers.append(gpt_answer)
#                 if "THE PEDESTRIAN IS NOT CROSSING" in gpt_answer:
#                     pedestrian_statuses.append("NOT CROSSING")
#                 elif "THE PEDESTRIAN IS CROSSING" in gpt_answer:
#                     pedestrian_statuses.append("CROSSING")

        
#         # Print the image file path along with corresponding human prompts, GPT answers, and pedestrian statuses
#         for human_prompt, gpt_answer, pedestrian_status in zip(human_prompts, gpt_answers, pedestrian_statuses):
#             IMAGE_FILE = image_file
#             QUERY = human_prompt
#             pedestrian_status = pedestrian_status
#             present = eval_model(pedestrian_status)
#             success = success + present

#     accuracy = (success/total)*100
#     print("Accuracy of model = ", accuracy)

import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import json


def image_parser(image_file):
    out = image_file.split(SEP)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(pedestrian_status):
    # Model
    disable_torch_init()
    print("Entered eval")

    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, MODEL_BASE, model_name
    )

    qs = QUERY
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "ped1"
    elif "mistral" in model_name.lower():
        conv_mode = "ped1"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "ped1"
    elif "v1" in model_name.lower():
        conv_mode = "ped1"
    elif "mpt" in model_name.lower():
        conv_mode = "ped1"
    else:
        conv_mode = "ped1"

    #if CONV_MODE is not None and conv_mode != CONV_MODE:
    #    print(
    #        "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
    #            conv_mode, CONV_MODE, CONV_MODE
    #        )
    #    )
    #else:
    CONV_MODE = conv_mode

    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(IMAGE_FILE)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    print(image_files)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if TEMPERATURE > 0 else False,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_beams=NUM_BEAMS,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    if pedestrian_status in outputs:
        present = 1
    else:
        present = 0
    print(outputs)
    return present


if __name__ == "__main__":
    # Define arguments directly within the code
    MODEL_PATH = "checkpoints0/llava-v1.5-7b-task-lora"
    MODEL_BASE = "liuhaotian/llava-v1.5-7b"
    IMAGE_FILE = "your_image.jpg"
    QUERY = "Your query text here"
    CONV_MODE = "ped1"
    SEP = ","
    TEMPERATURE = 0.2
    TOP_P = None
    NUM_BEAMS = 1
    MAX_NEW_TOKENS = 512

    with open('./playground/data/overall_val_prompts_fm_rev.json', 'r') as file:
        data = json.load(file)

    success = 0
    total = 0

    for entry in data:
        total = total + 1
        image_file = entry['image']
        
        human_prompts = []
        gpt_answers = []
        pedestrian_statuses = []
        

        conversations = entry['conversations']
        
        for conversation in conversations:
            if conversation['from'] == 'human':
                human_prompts.append(conversation['value'])
            elif conversation['from'] == 'gpt':
                # Extract and store the GPT answer
                gpt_answer = conversation['value']
                gpt_answers.append(gpt_answer)
                # Check for pedestrian status in the GPT answer
                if "THE PEDESTRIAN IS NOT CROSSING" in gpt_answer:
                    pedestrian_statuses.append("NOT CROSSING")
                elif "THE PEDESTRIAN IS CROSSING" in gpt_answer:
                    pedestrian_statuses.append("CROSSING")

         
        for human_prompt, gpt_answer, pedestrian_status in zip(human_prompts, gpt_answers, pedestrian_statuses):
            IMAGE_FILE = image_file
            QUERY = human_prompt
            pedestrian_status = pedestrian_status
            present = eval_model(pedestrian_status)
            success = success + present

    accuracy = (success/total)*100
    print("Accuracy of model = ", accuracy)

    

