

import pandas as pd

import torch


def load_raw_texts(dataset):
    raw_texts_path = f'dataset/{dataset}/raw_texts.pt'
    raw_texts = torch.load(raw_texts_path)
    return raw_texts

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}'

def get_detailed_example(task_description: str, query: str, response: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}\n<response>{response}'

class Prompt():
    def __init__(self, texts2encode, labels):
        # initialization
        self.texts2encode = texts2encode
        self.labels = labels

        self.num_texts = len(self.texts2encode)
        self.num_labels = len(self.labels)
        # task: the meta information description of task
        # task = ''
        # examples: whether do few-shot or not, default: do not use examples
        # examples = [{'instruct', 'query', 'response'}]
        # queries: after the task, say the query
        # queries = [get_detailed_instruct(task, '')]
        # documents: the text to encode
        # documents = ['']

    def prepare_prompts(self, version):
        # version 
        # primary: nothing
        # class_aware: know the class
        if version == 'primary':
            self.task = self.get_primary_task()
            self.examples_prefix = ''
            self.queries = [get_detailed_instruct(self.task, '') + '\n' + self.texts2encode[i] for i in range(self.num_texts)]
        
        elif version == 'class_aware':
            self.task = self.get_class_aware_task()
            class_description = ''
            for i in range(self.num_labels):
                class_description += self.labels[i] + '\n'
            self.task += class_description
            self.examples_prefix = ''
            self.queries = [get_detailed_instruct(self.task, '') + '\n' + self.texts2encode[i] for i in range(self.num_texts)]

    def get_class_aware_task(self):
        raise NotImplementedError
    
    def get_primary_task(self):
        raise NotImplementedError


class Prompt_citeseer(Prompt):
    def __init__(self, texts2encode, labels):
        super().__init__(texts2encode, labels)

    def get_class_aware_task(self):
        return 'Given the description or opening text of scientific publications, classify it into one of the following 6 classes: \n'
    def get_primary_task(self):
        return 'Encode the description or opening text of scientific publications: \n'

class Prompt_cora(Prompt):
    def __init__(self, texts2encode, labels):
        super().__init__(texts2encode, labels)
    def get_class_aware_task(self):
        return 'Given the opening text of machine learning papers, classify it into one of the following 7 classes: \n'
    def get_primary_task(self):
        return 'Encode the text of machine learning papers: \n'

class Prompt_pubmed(Prompt):
    def __init__(self, texts2encode, labels):
        super().__init__(texts2encode, labels)
        self.labels = [
            'Diabetes Mellitus Experimental (animal models, cell-based experiments)',
            'Diabetes Mellitus Type 1 (autoimmune condition causing absolute insulin deficiency)',
            'Diabetes Mellitus Type 2 (insulin resistance and a progressive decline in insulin production)']
    def get_class_aware_task(self):
        return 'Given the title and abstract of scientific publications, classify it into one of the following 3 classes: \n'
    def get_primary_task(self):
        return 'Encode the title and abstract of scientific publications: \n'
        

class Prompt_wikics(Prompt):
    def __init__(self, texts2encode, labels):
        super().__init__(texts2encode, labels)
        # pre-process the text. remove some useless texts
        char1 = 'feature node. wikipedia entry name:'
        char2 = 'entry content:'
        new_char1 = 'entry:'
        new_char2 = 'content:'
        processed_list = [s.replace(char1, new_char1).replace(char2, new_char2) for s in self.texts2encode]
        self.texts2encode = processed_list
    def get_class_aware_task(self):
        return 'Given the entry and content of wikipedia, classify it into one of the following 10 classes: \n'
    def get_primary_task(self):
        return 'Encode the entry and content of wikipedia: \n'
    
class Prompt_bookhis(Prompt):
    def __init__(self, texts2encode, labels):
        super().__init__(texts2encode, labels)
    def get_class_aware_task(self):
        return 'Given the description or title of the book, classify it into one of the following 12 classes: \n'
    def get_primary_task(self):
        return 'Encode the description or title of the book: \n'
    
class Prompt_bookchild(Prompt):
    def __init__(self, texts2encode, labels):
        super().__init__(texts2encode, labels)
    def get_class_aware_task(self):
        return 'Given the description or title of the child literature, classify it into one of the following 24 classes: \n'
    def get_primary_task(self):
        return 'Encode the description or title of the child literature: \n'

class Prompt_sportsfit(Prompt):
    def __init__(self, texts2encode, labels):
        super().__init__(texts2encode, labels)
        char1 = 'The title of the item in this Sports & Fitness category is'
        new_char1 = 'The title is'
        processed_list = [s.replace(char1, new_char1) for s in self.texts2encode]
        self.texts2encode = processed_list
    def get_class_aware_task(self):
        return 'Given the title of a good in sports & fitness, classify it into one of the following 13 classes: \n'
    def get_primary_task(self):
        return 'Encode the title of a good in sports & fitness: \n'

class Prompt_cornell(Prompt):
    def __init__(self, texts2encode, labels):
        super().__init__(texts2encode, labels)
    def get_class_aware_task(self):
        return 'Given a webpage text, classify it into one of the following 5 classes: \n'
    def get_primary_task(self):
        return 'Encode the webpage text: \n'

class Prompt_wisconsin(Prompt):
    def __init__(self, texts2encode, labels):
        super().__init__(texts2encode, labels)
    def get_class_aware_task(self):
        return 'Given a webpage text, classify it into one of the following 5 classes: \n'
    def get_primary_task(self):
        return 'Encode the webpage text: \n'

class Prompt_washington(Prompt):
    def __init__(self, texts2encode, labels):
        super().__init__(texts2encode, labels)
    def get_class_aware_task(self):
        return 'Given a webpage text, classify it into one of the following 5 classes: \n'
    def get_primary_task(self):
        return 'Encode the webpage text: \n'

class Prompt_texas(Prompt):
    def __init__(self, texts2encode, labels):
        super().__init__(texts2encode, labels)
    def get_class_aware_task(self):
        return 'Given a webpage text, classify it into one of the following 5 classes: \n'
    def get_primary_task(self):
        return 'Encode the webpage text: \n'
        
