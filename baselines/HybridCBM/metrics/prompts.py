import base64

import pandas as pd
from click import prompt


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_class_from_dataset(dataset):
    if dataset == "awa2":
        x = pd.read_csv(
            f"/DATA/Animals_with_Attributes2/concepts/concepts.csv"
        )
    elif dataset == "cifar100":
        x = pd.read_csv(
            f"/DATA/cifar100_images/concepts/concepts.csv"
        )
    elif dataset == "inet100":
        x = pd.read_csv(
            f"/DATA/inet100/concepts/concepts.csv"
        )
    # example = x.iloc[:3]
    x = x[["class", "label"]].drop_duplicates()
    return x["class"].tolist(), x["label"].to_list()


def get_class_prompt(batch_size, delimiter=","):
    prompts = [
        {
            "role": "system",
            "content": f"You are an expert binary concept classifier. You will receive pairs of (concept, class), "
            f"and your task is to determine whether the concept describes or has any relationship with the "
            f"given class. For each pair, respond with 'yes' or 'no'. You will receive {batch_size} pairs at once. "
            f"Please reply with {batch_size} answers in a single line, using 'yes' or 'no', separated by "
            f"'{delimiter}' (e.g., {', '.join((batch_size * ['yes', 'no'])[:batch_size])}).",
        }
    ]
    return prompts


def get_prompts_from_dataset(class_name, delimiter=",", batch_size=2):
    prompts = [
        {
            "role": "system",
            "content": f"You are an expert binary concept classifier, whether the given concept has any form of "
            f"relationship with any of the predefined classes. A relationship can include similarity, "
            f"subordination, purpose, common attributes, or any other form of connection. \n"
            f"For each concept: if it has any relationship with any of the classes, response 'yes', "
            f"otherwise, response 'no'. You'll receive {batch_size} input concepts at once, separated by '{delimiter}'. "
            f"And you should also reply with {batch_size} answers in a single line, separated by '{delimiter}'. "
            f"(e.g., {', '.join((batch_size * ['yes', 'no'])[:batch_size])}). \n"
            f"The predefined classes are as follows:  {', '.join(class_name)}.",
        },
    ]
    return prompts


def get_vlm_prompts(image_path, concepts, delimiter="$"):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "You are an expert binary concept classifie, able to determine whether the given concept "
                    "has any form of relationship with the given image. A relationship can include similarity, "
                    f"subordination, purpose, common attributes, or any other form of connection.\n"
                    "For each concept: if it has any relationship with the image, response 'yes', "
                    f"otherwise, response 'no'. You'll receive {len(concepts)} input concepts at once, separated "
                    f"by '{delimiter}'. And you should also reply with {len(concepts)} answers in a single line, "
                    f"separated by '{delimiter}'. (e.g., {f'{delimiter} '.join((len(concepts) * ['yes', 'no'])[:len(concepts)])}). \n"
                    f"The concepts are as follows: {delimiter.join(concepts)}{delimiter}. The image is shown below.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                    },
                },
            ],
        }
    ]
