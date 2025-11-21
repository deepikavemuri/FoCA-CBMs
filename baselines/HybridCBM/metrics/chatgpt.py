# -*- encoding: utf-8 -*-

import time
import re
import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from .prompts import get_prompts_from_dataset, get_class_from_dataset, get_vlm_prompts, get_class_prompt


class ChatGPT:
    def __init__(self, api_base, api_key, model,
                 prompts=None,
                 conversation_track=False,
                 temperature=0,
                 top_p=1,
                 stream=False
                 ):
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            default_headers={"x-foo": "true"},
        )

        self.conversation_track = conversation_track
        self.conversations = {}
        self.prompts = prompts
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.stream = stream

    def __call__(self, text_message=None, prompts=None, user_id=None):
        """
        Make remember all the conversation
        :param old_model: Open AI model
        :param user_id: telegram user id
        :param text_message: text message
        :return: str
        """
        if text_message is not None:
            text_message = [{"role": "user", "content": text_message}]

        if not self.conversation_track:
            # Generate response

            return self.generate_response_chatgpt(text_message, prompts=prompts)

        conversation_history, gpt_responses = [], []
        # Get the last 10 conversations and responses for this user
        user_conversations = self.conversations.get(user_id, {'conversations': [], 'responses': []})
        user_messages = user_conversations['conversations'] + [text_message]
        gpt_responses = user_conversations['responses']

        # Store the updated conversations and responses for this user
        self.conversations[user_id] = {'conversations': user_messages, 'responses': gpt_responses}

        # Construct the full conversation history in the user:assistant, " format
        for i in range(min(len(user_messages), len(gpt_responses))):
            conversation_history.append(user_messages[i])
            conversation_history.append(gpt_responses[i])

        # Add last prompt
        conversation_history.append(text_message)

        # Generate response
        response = self.generate_response_chatgpt(conversation_history, prompts=prompts)

        # Add the response to the user's responses
        gpt_responses.append(response)
        # Store the updated conversations and responses for this user
        self.conversations[user_id] = {'conversations': user_messages, 'responses': gpt_responses}
        return response

    def generate_response_chatgpt(self, message_list, prompts=None):
        if prompts is None:
            prompts = self.prompts
        if prompts is None:
            prompts = []
        if message_list is None:
            message_list = []
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompts + message_list,
                stream=self.stream,
                temperature=self.temperature,
                frequency_penalty=0,
                presence_penalty=0,
                top_p=self.top_p
            )
        except Exception as e:
            time.sleep(1)
            print(e)
            return self.generate_response_chatgpt(message_list, prompts)
        if self.stream:
            return response
        return response.choices[0].message.content.strip()


class LLMEvaluation:
    def __init__(self, api_base, api_key, dataset, model, delimiter=','):
        super().__init__()
        classes, labels = get_class_from_dataset(dataset)
        self.dataset = dataset
        self.llm = ChatGPT(api_base, api_key, model)
        self.class2label = {c: l for c, l in zip(classes, labels)}
        self.label2class = {l: c for c, l in zip(classes, labels)}
        self.delimiter = delimiter

    def concept_alignment(self, concept):
        concept = [re.sub(r'[^a-zA-Z ,]', '', c).strip() for c in concept]
        prompts = get_prompts_from_dataset(
            class_name=list(self.class2label.keys()),
            delimiter=self.delimiter,
            batch_size=len(concept))
        result = self.llm(f' {self.delimiter} '.join(concept), prompts=prompts).strip().split(self.delimiter)
        while '' in result:
            result.remove('')
        while len(result) != len(concept):
            print(f'retrying. {concept} : {result}')
            result = self.concept_alignment(concept)
        return result

    def concept_semantic(self, concept, class_name=None):
        if class_name is None:
            concept = [re.sub(r'[^a-zA-Z ,]', '', c).strip() for c in concept]
            inputs = [f'({c}, {class_name})' for c in concept]
        else:
            inputs = [[re.sub(r'[^a-zA-Z ,]', '', c[0]).strip(), c[1]] for c in concept]
        prompts = get_class_prompt(batch_size=len(concept), delimiter=',')
        result = self.llm(f'\n'.join(inputs), prompts=prompts).strip().split(',')
        while '' in result:
            result.remove('')
        while len(result) != len(concept):
            print(f'retrying. {concept} : {result}')
            result = self.concept_semantic(concept, class_name)
        return result

    def __call__(self, concepts, class_name=None, to_label=False, batch_size=2):
        results = []
        for i in tqdm(range(0, len(concepts), batch_size)):
            concept = concepts[i:i + batch_size]
            if class_name is None:
                result = self.concept_alignment(concept)
            else:
                result = self.concept_semantic(concept, class_name)
            results.extend(result)
        if to_label:
            results = [self.class2label.get(r, -1) for r in results]
        return results

    def multi_evaluate(self, concepts, n_jobs=2, to_label=False):
        import mpire as mpi
        with mpi.WorkerPool(n_jobs=n_jobs) as pool:
            results = pool.map(self.__call__, concepts)
        return results


class VLMEvaluation:
    def __init__(self, api_base, api_key, model='gpt-4o', delimiter=';'):
        super().__init__()
        self.vlm = ChatGPT(api_base, api_key, model)
        self.delimiter = delimiter

    def evaluate(self, image_path, concepts):
        concepts = [re.sub(r'[^a-zA-Z ,]', '', c).strip().replace('  ', ' ') for c in concepts]
        prompts = get_vlm_prompts(image_path=image_path,
                                  concepts=concepts,
                                  delimiter=self.delimiter)
        result = list(map(lambda x: x.strip(), self.vlm(prompts=prompts).split(self.delimiter)))
        while '' in result:
            result.remove('')
        while len(result) != len(concepts):
            print(f'retrying. {concepts} : {result}')
            if len(result) < len(concepts):
                result = self.evaluate(image_path, concepts)
            else:
                result = result[:len(concepts)]
        return result


def compute_semantics(path, force=False):
    acc_path = path.replace('.csv', '_LLM_Data_acc.csv')
    if os.path.exists(acc_path) and not force:
        all_acc = pd.read_csv(acc_path)
        return all_acc.iloc[-1]['acc']
    concepts = pd.read_csv(path)['concept'].unique()
    evaluator = LLMEvaluation(
        api_base='<your base>',
        api_key='<your key>',
        dataset='CUB',
        model='gpt-3.5-turbo'
    )
    all_acc = []
    for concept in tqdm(range(0, len(concepts), 10)):
        concept = concepts[concept:concept + 10]
        result = evaluator.concept_alignment(concept)
        result = [int(r.strip() != 'no') for r in result]
        all_acc.extend(result)
    all_acc = pd.DataFrame([{'acc': sum(all_acc) / len(all_acc), 'class': 'mean'}])
    all_acc.to_csv(path.replace('.csv', '_LLM_Data_acc.csv'), index=False)
    return all_acc.iloc[-1]['acc']


def compute_precision_k(path, k=5, force=False):
    images = pd.read_csv('datasets/CUB/sampled_ViT-L-14_images/images.csv')
    acc_path = path.replace('.csv', '_LLM_img_acc.csv')
    result_path = path.replace('.csv', '_LLM_img_result.csv')
    if os.path.exists(acc_path) and not force:
        all_acc = pd.read_csv(acc_path)
        print(all_acc.iloc[-1])
        return all_acc.iloc[-1]['acc']
    concepts = \
        pd.read_csv(path).groupby('idx').apply(lambda x: x.drop_duplicates('concept').iloc[:k]).reset_index(drop=True)[
            ['concept', 'idx']]
    evaluator = VLMEvaluation(
        api_base='<your base>',
        api_key='<your key>',
        model='gpt-4o'
    )
    results = []
    all_acc = []

    for label, path in tqdm(list(zip(images['label'], images['path']))):
        concept = concepts.loc[concepts['idx'] == label, 'concept'].to_list()
        result = evaluator.evaluate(path, concept)
        result = [int(r.strip() != 'no') for r in result]
        for i in range(0, len(result)):
            results.append({'path': path, 'concept': concept[i], 'result': result[i]})
        all_acc.append({'acc': sum(result) / k, 'class': label})
    results = pd.DataFrame(results)
    results.to_csv(result_path, index=False)
    all_acc = pd.DataFrame(all_acc)
    all_acc = pd.concat([all_acc, pd.DataFrame([{'acc': all_acc['acc'].mean(),
                                                 'class': 'mean'}])]).reset_index(drop=True)
    all_acc.to_csv(acc_path, index=False)
    print(all_acc.iloc[-1])
    return all_acc.iloc[-1]['acc']
