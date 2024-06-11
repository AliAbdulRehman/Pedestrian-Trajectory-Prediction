# # preprocess data
# import json
# import re
# with open('./playground/data/overall_val_prompts_aa.json', 'r') as file:
#     data = json.load(file)
# query_image = []
# query_prompt = []
# ground_truth = []
# ground_truth_coordinate = []
# ground_truth_label = []

# for entry in data:
#     query_image.append(entry['image'])
#     conversations = entry['conversations']
#     for conversation in conversations:
#         if conversation['from'] == 'human':
#             query_prompt.append(conversation['value'][8:]) # remove <image> at the beginning of the prompts
#         elif conversation['from'] == 'gpt':
#             ground_truth.append(conversation['value'])
#             if 'NOT CROSSING' in conversation['value']:
#                 ground_truth_label.append(0)
#             else:
#                 ground_truth_label.append(1)
#             regular_expression = re.findall(r'\[\[.*?\]\]', conversation['value'])
#             ground_truth_coordinate.append(eval(regular_expression[0]))
# preprocess data
import json
import re
with open('./playground/data/overall_val_prompts_aa.json', 'r') as file:
    data = json.load(file)
query_image = []
query_prompt = []
ground_truth = []
ground_truth_coordinate = []
ground_truth_label = []
count=0
for entry in data:
    regular_expression = []
    count=count+1
    print(count)
    query_image.append(entry['image'])
    conversations = entry['conversations']
    for conversation in conversations:
        if conversation['from'] == 'human':
            query_prompt.append(conversation['value'][8:]) # remove <image> at the beginning of the prompts
        elif conversation['from'] == 'gpt':
            ground_truth.append(conversation['value'])
            if 'not-crossing' in conversation['value']:
                ground_truth_label.append(0)
            else:
                ground_truth_label.append(1)
            regular_expression = re.findall(r'\[\[.*?\]\]', conversation['value'])
            # print(regular_expression[0])
            if regular_expression != []:
                ground_truth_coordinate.append(eval(regular_expression[0]))
            else:
                ground_truth_coordinate.append(eval("[]"))
#print(query_image)
#print(query_prompt)
#print(ground_truth)
#print(ground_truth_coordinate)
#print(ground_truth_label)

from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import *

MODEL_PATH = "checkpoints11/llava-v1.5-7b-task-lora"
MODEL_BASE = "liuhaotian/llava-v1.5-7b"
CONV_MODE = None
SEP = ","
TEMPERATURE = 0
TOP_P = None
NUM_BEAMS = 1
MAX_NEW_TOKENS = 512

def eval_model(args, tokenizer, model, image_processor):
    disable_torch_init()
    
    qs = args.query
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

    if "llama-2" in args.model_name.lower():
        conv_mode = "ped1"
    elif "mistral" in args.model_name.lower():
        conv_mode = "ped1"
    elif "v1.6-34b" in args.model_name.lower():
        conv_mode = "ped1"
    elif "v1" in args.model_name.lower():
        conv_mode = "ped1"
    elif "mpt" in args.model_name.lower():
        conv_mode = "ped1"
    else:
        conv_mode = "ped1"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

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
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

tokenizer, model, image_processor, context_len = load_pretrained_model(
model_path=MODEL_PATH,
model_base=MODEL_BASE,
model_name=get_model_name_from_path(MODEL_PATH))

model_outputs = []
for i in range(len(query_image)):
    args = type('Args', (), {
                "model_name": get_model_name_from_path(MODEL_PATH),
                "query": query_prompt[i],
                "conv_mode": CONV_MODE,
                "image_file": query_image[i],
                "sep": SEP,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "num_beams": NUM_BEAMS,
                "max_new_tokens": MAX_NEW_TOKENS
            })()    
    model_outputs.append(eval_model(args, tokenizer, model, image_processor))




# process model output
output_coordinate = []
output_label = []
for output in model_outputs:
    if 'not-crossing' in output:
        output_label.append(0)
    else:
        output_label.append(1)
    regular_expression = re.findall(r'\[\[.*?\]\]', output)
    output_coordinate.append(eval(regular_expression[0]))
#print(model_outputs)
#print(output_coordinate)
#print(output_label)




# calculate metrices
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
accuracy = accuracy_score(ground_truth_label, output_label)
print("Accuracy:", accuracy)
f1 = f1_score(ground_truth_label, output_label)
print("F1 Score:", f1)
ground_truth_coordinate = np.array(ground_truth_coordinate).flatten()
output_coordinate = np.array(output_coordinate).flatten()
mse = mean_squared_error(ground_truth_coordinate, output_coordinate)
print("RMSE:", np.sqrt(mse))
