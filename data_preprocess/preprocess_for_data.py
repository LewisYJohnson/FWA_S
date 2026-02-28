# -*- coding : utf-8 -*-
import re

from torch.utils import data
import torch

from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from transformers import RobertaTokenizer, RobertaModel

from constant import hacred_ner_labels_constant, hacred_rel_labels_constant, hacred_rel_to_question, get_labelmap, \
    scierc_ner_labels_constant, scierc_rel_labels_constant, scierc_rel_to_question, \
    _2017t10_rel_labels_constant, _2017t10_ner_labels_constant, _2017t10_rel_to_question, \
    ace05_rel_labels_constant, ace05_ner_labels_constant, ace05_rel_to_question, \
    conll04_ner_labels_constant, conll04_rel_labels_constant, conll04_rel_to_question, \
    ADE_ner_labels_constant, ADE_rel_labels_constant, ADE_rel_to_question
# from data_config import ArgumentParser

import numpy as np
import random
import json
import logging
from tqdm import tqdm
import pickle
import os
import sys

'''
scierc 12
2017t10 24
2022t12 24
'''

class ProcessedDataset(data.DataLoader):
    def __init__(self, config_params, json_input, dataset_category, output_dir, processing_stage):
        self.processing_stage = processing_stage
        self.config_params = config_params
        self.dataset_category = dataset_category
        self.json_input = json_input

        self.pre_train_model = self.config_params["pretrained_model"]
        if self.pre_train_model != "roberta-base":
            self.text_tokenizer = BertTokenizer.from_pretrained(self.pre_train_model)
        else:
            self.text_tokenizer = RobertaTokenizer.from_pretrained(self.pre_train_model)

        self.dataset_task = self.config_params["task"]
        self.enhanced_star = self.config_params["star"]

        if self.dataset_task == "scierc":
            ner_label_mapping, ner_id_mapping = generate_label_mapping(scierc_ner_labels_constant)
        else:
            ner_label_mapping, ner_id_mapping = eval("generate_label_mapping("+self.dataset_task+"_ner_labels_constant)")
        if os.path.exists(output_dir + self.dataset_task + "_preprocessed_" + self.dataset_category + "_data.pkl"):
            prepared_records = pickle.load(open(output_dir + self.dataset_task + "_preprocessed_" + self.dataset_category +
                                                 "_data.pkl", "rb"))
            print("Loaded preprocessed records successfully!")
        else:
            prepared_records = self.prepare_records(ner_label_mapping)
            pickle.dump(prepared_records, open(output_dir + self.dataset_task + "_preprocessed_" + self.dataset_category
                                                + "_data.pkl", "wb"))
            print("Saved preprocessed records successfully!")

        if os.path.exists(output_dir + self.dataset_task + "_all_" + self.dataset_category + "_entries.pkl"):
            self.complete_entries = self.transform_all_records(prepared_records)
            self.complete_entries = pickle.load(open(output_dir + self.dataset_task + "_all_" + self.dataset_category +
                                                "_entries.pkl", "rb"))
            print("Loaded all dataset entries successfully!")
        else:
            self.complete_entries = self.transform_all_records(prepared_records)
            pickle.dump(self.complete_entries, open(output_dir + self.dataset_task + "_all_" + self.dataset_category +
                                               "_entries.pkl", "wb"))
            print("Saved all dataset entries successfully!")
        print("Dataset preparation completed!")

    def __len__(self):
        return len(self.complete_entries)

    def __getitem__(self, idx):
        return self.complete_entries[idx]

    def formulate_template_question(self, rel_key):
        question_words = []
        question_words.append(self.text_tokenizer.mask_token)

        rel_template_mapping = eval(self.dataset_task + '_rel_to_question')

        question_words += self.text_tokenizer.tokenize(rel_template_mapping[rel_key])
        question_words.append(self.text_tokenizer.mask_token)
        question_words.append(self.text_tokenizer.sep_token)
        question_ids = self.text_tokenizer.convert_tokens_to_ids(question_words)
        return question_ids

    def prepare_decoder_input(self, data_sample):
        grouped_relations = {}
        for relation_group in data_sample["relations"].values():
            for relation_item in relation_group:
                rel_type = relation_item[2]
                entity_pair = [relation_item[0], relation_item[1]]
                if rel_type not in grouped_relations.keys():
                    grouped_relations[rel_type] = []
                grouped_relations[rel_type].append(entity_pair)

        rel_labels_const = eval(self.dataset_task + '_rel_labels_constant')

        relation_label_to_id, relation_id_to_label = generate_label_mapping(rel_labels_const)

        for relation_label in rel_labels_const:
            if relation_label not in grouped_relations.keys():
                grouped_relations[relation_label] = []

        if self.config_params["sorted_entity"]:
            for relation, pairs in grouped_relations.items():
                grouped_relations[relation] = sorted(pairs, key=lambda x: 1000000 * x[0] + x[1])
        else:
            for relation, pairs in grouped_relations.items():
                random.shuffle(pairs)

        if self.config_params["sorted_relation"]:
            relation_pairs = sorted(grouped_relations.items(), key=lambda x: x[0])
        else:
            relation_pairs = sorted(grouped_relations.items(), key=lambda x: x[0])
            random.shuffle(relation_pairs)

        relation_entries = relation_pairs

        relation_id_list = []
        question_inputs = []
        answer_inputs = []
        mask_position_list = []
        token_type_identifiers = []
        for relation_label, entity_pairs in relation_entries:
            prepared_answers = []
            question_template = self.formulate_template_question(relation_label) * self.config_params["duplicate_questions"]
            question_template[-1] = self.text_tokenizer.convert_tokens_to_ids(".")

            if self.config_params["token_type_ids"]:
                token_type_ids = self.generate_token_type_ids(question_template)
            else:
                token_type_ids = [0] * len(question_template)

            mask_positions = np.argwhere(np.array(question_template) ==
                                   self.text_tokenizer.convert_tokens_to_ids(self.text_tokenizer.mask_token))
            mask_positions = mask_positions.squeeze().tolist()
            prepared_answers.extend(entity_pairs)
            if len(prepared_answers) > self.config_params["duplicate_questions"]:
                prepared_answers = prepared_answers[:self.config_params["duplicate_questions"]]
            for i in range(self.config_params["duplicate_questions"] - len(entity_pairs)):
                prepared_answers.append([-1, -1])

            relation_id_list.append([relation_label_to_id[relation_label]])
            question_inputs.append(question_template)
            answer_inputs.extend([prepared_answers])
            mask_position_list.append(mask_positions)
            token_type_identifiers.append(token_type_ids)

        answer_inputs = np.array(answer_inputs)
        answer_inputs = answer_inputs.reshape(answer_inputs.shape[0], -1).tolist()
        return relation_id_list, question_inputs, answer_inputs, mask_position_list, token_type_identifiers

    def prepare_SPN_input(self, data_sample):
        rel_labels_const = eval(self.dataset_task + '_rel_labels_constant')

        relation_label_to_id, relation_id_to_label = generate_label_mapping(rel_labels_const)
        relation_ids = []
        subject_entity_positions = []
        object_entity_positions = []
        for relation_group in data_sample["relations"].values():
            for relation_item in relation_group:
                relation_ids.append(relation_label_to_id[relation_item[2]])
                subject_entity_positions.append(relation_item[0])
                object_entity_positions.append(relation_item[1])
        return relation_ids, subject_entity_positions, object_entity_positions

    def generate_token_type_ids(self, question_input):
        token_type_ids = []
        is_first_segment = True
        for token_id in question_input:
            if is_first_segment:
                token_type_ids.append(0)
            else:
                token_type_ids.append(1)
            if token_id == self.text_tokenizer.convert_tokens_to_ids(self.text_tokenizer.sep_token):
                is_first_segment = not is_first_segment
        return token_type_ids

    def convert_single_entry(self, data_sample):
        id_to_start = []
        id_to_end = []
        tokenized_words = []
        transformed_samples = []
        entity_span_masks = []

        tokenized_words.append(self.text_tokenizer.cls_token)
        for word in data_sample["sentences"]:
            id_to_start.append(len(tokenized_words))
            sub_words = self.text_tokenizer.tokenize(word)
            tokenized_words += sub_words
            id_to_end.append(len(tokenized_words)-1)


        input_ids = self.text_tokenizer.convert_tokens_to_ids(tokenized_words)
        span_entries = [[id_to_start[span[0]], id_to_end[span[1]], span[2]] for span in data_sample["spans"]]
        relation_ids, subject_entity_positions, object_entity_positions = self.prepare_SPN_input(data_sample)
        relation_labels_list, input_decoder_questions, input_decoder_answers, input_mask_positions, \
        question_token_types = self.prepare_decoder_input(data_sample)

        neg_sample_prob = None
        if self.processing_stage == "1":
            neg_sample_prob = self.config_params["negative_probability"]
        if self.processing_stage == "2":
            neg_sample_prob = 0

        for rel_id, question, answer, mask_pos, token_type in zip(relation_labels_list, input_decoder_questions,
                                                                 input_decoder_answers, input_mask_positions,
                                                                 question_token_types):
            rand_val = random.random()
            if sum(answer) == -len(answer) and rand_val < neg_sample_prob:
                transformed_sample = {}
                transformed_sample["question_template"] = torch.tensor(question)
                transformed_sample["template_relation"] = torch.tensor(rel_id)
                transformed_sample["answer_pairs"] = torch.tensor(answer)
                transformed_sample["mask_position"] = torch.tensor(mask_pos)
                transformed_sample["token_type"] = torch.tensor(token_type)

                transformed_sample["input_ids"] = torch.tensor(input_ids)
                transformed_sample["span_info"] = torch.tensor(span_entries)
                transformed_sample["span_labels"] = torch.tensor(data_sample["spans_label"])
                transformed_sample["doc_id"] = data_sample["doc_key"]

                transformed_sample["relation_ids"] = torch.tensor(relation_ids)
                transformed_sample["subject_pos"] = torch.tensor(subject_entity_positions)
                transformed_sample["object_pos"] = torch.tensor(object_entity_positions)
                transformed_sample["span_masks"] = entity_span_masks
                transformed_samples.append(transformed_sample)
            elif sum(answer) != -len(answer):
                transformed_sample = {}
                transformed_sample["question_template"] = torch.tensor(question)
                transformed_sample["template_relation"] = torch.tensor(rel_id)
                transformed_sample["answer_pairs"] = torch.tensor(answer)
                transformed_sample["mask_position"] = torch.tensor(mask_pos)
                transformed_sample["token_type"] = torch.tensor(token_type)

                transformed_sample["input_ids"] = torch.tensor(input_ids)
                transformed_sample["span_info"] = torch.tensor(span_entries)
                transformed_sample["span_labels"] = torch.tensor(data_sample["spans_label"])
                transformed_sample["doc_id"] = data_sample["doc_key"]

                transformed_sample["relation_ids"] = torch.tensor(relation_ids)
                transformed_sample["subject_pos"] = torch.tensor(subject_entity_positions)
                transformed_sample["object_pos"] = torch.tensor(object_entity_positions)
                transformed_sample["span_masks"] = entity_span_masks
                transformed_samples.append(transformed_sample)
            else:
                pass
        if len(transformed_samples) == 0:
            rand_index = int(random.random() * len(input_decoder_questions))
            transformed_sample = {}
            transformed_sample["question_template"] = torch.tensor(input_decoder_questions[rand_index])
            transformed_sample["template_relation"] = torch.tensor(relation_labels_list[rand_index])
            transformed_sample["answer_pairs"] = torch.tensor(input_decoder_answers[rand_index])
            transformed_sample["mask_position"] = torch.tensor(input_mask_positions[rand_index])
            transformed_sample["token_type"] = torch.tensor(question_token_types[rand_index])

            transformed_sample["input_ids"] = torch.tensor(input_ids)
            transformed_sample["span_info"] = torch.tensor(span_entries)
            transformed_sample["span_labels"] = torch.tensor(data_sample["spans_label"])
            transformed_sample["doc_id"] = data_sample["doc_key"]

            transformed_sample["relation_ids"] = torch.tensor(relation_ids)
            transformed_sample["subject_pos"] = torch.tensor(subject_entity_positions)
            transformed_sample["object_pos"] = torch.tensor(object_entity_positions)
            transformed_sample["span_masks"] = entity_span_masks
            transformed_samples.append(transformed_sample)
        return transformed_samples

    def transform_all_records(self, preprocessed_data):
        all_entries = []
        for record in tqdm(preprocessed_data):
            transformed_records = self.convert_single_entry(record)
            all_entries.extend(transformed_records)
        return all_entries

    def prepare_records(self, ner_label_mapping):
        with open(self.json_input, 'r', encoding='utf8') as f:
            documents = json.load(f)
        max_span_count = 0
        total_spans = 0
        finalized_records = []
        for doc_idx, record in tqdm(enumerate(documents)):
            finalized_record = {}
            words = self.normalize_text(record["sentText"]).split(" ")
            finalized_record["doc_key"] = "sentence" + str(doc_idx)
            finalized_record["sentences"] = words

            relation_triples = record["relationMentions"]
            finalized_record["ner"] = {}
            for triple in relation_triples:
                subject_entity = self.normalize_text(triple["em1Text"]).split(" ")
                object_entity = self.normalize_text(triple["em2Text"]).split(" ")
                if not self.config_params["is_chinese"]:
                    subj_start, subj_end = self.locate_entity(subject_entity, words)
                    obj_start, obj_end = self.locate_entity(object_entity, words)
                else:
                    subj_start, subj_end = triple['ent1_loc'][0], triple['ent1_loc'][1] - 1
                    obj_start, obj_end = triple['ent2_loc'][0], triple['ent2_loc'][1] - 1

                finalized_record["ner"][(subj_start, subj_end)] = (words[subj_start: subj_end + 1], "entity")
                finalized_record["ner"][(obj_start, obj_end)] = (words[obj_start: obj_end + 1], "entity")
                if obj_end - obj_start >= self.config_params["max_span_length"]:
                    print("\n" + "Note: entity exceeds max_span_length: "
                          + str(self.config_params["max_span_length"]), ". Entity: ( "
                          + " ".join(words[obj_start: obj_end + 1]) + " ). Length: "
                          + str(obj_end - obj_start) + " .")

            span_to_id = {}
            finalized_record['spans'] = []
            finalized_record['spans_label'] = []
            for i in range(len(words)):
                for j in range(i, min(len(words), i + self.config_params["max_span_length"])):
                    start_idx = i
                    end_idx = j
                    finalized_record['spans'].append((start_idx, end_idx, j - i + 1))
                    span_to_id[(start_idx, end_idx)] = len(finalized_record['spans']) - 1
                    if (start_idx, end_idx) not in finalized_record["ner"].keys():
                        finalized_record['spans_label'].append(0)
                    else:
                        label_name = finalized_record["ner"][(start_idx, end_idx)][1]
                        finalized_record['spans_label'].append(ner_label_mapping[label_name])

            max_span_count = max(len(finalized_record['spans_label']), max_span_count)
            total_spans = total_spans + len(finalized_record['spans_label'])
            finalized_record["relations"] = {}

            for relation in relation_triples:
                subject_entity = self.normalize_text(relation["em1Text"]).split(" ")
                object_entity = self.normalize_text(relation["em2Text"]).split(" ")
                if not self.config_params["is_chinese"]:
                    e1_start, e1_end = self.locate_entity(subject_entity, words)
                    e2_start, e2_end = self.locate_entity(object_entity, words)
                else:
                    e1_start, e1_end = relation['ent1_loc'][0], relation['ent1_loc'][1] - 1
                    e2_start, e2_end = relation['ent2_loc'][0], relation['ent2_loc'][1] - 1

                if e1_end - e1_start >= self.config_params["rel_span"] or e2_end - e2_start >= self.config_params["rel_span"]:
                    if e1_end - e1_start >= self.config_params["rel_span"]:
                        print("subject entity length %d" % (e1_end - e1_start))
                    if e2_end - e2_start >= self.config_params["rel_span"]:
                        print("object entity length %d" % (e2_end - e2_start))
                else:
                    e1_span_id = span_to_id[(e1_start, e1_end)]
                    e2_span_id = span_to_id[(e2_start, e2_end)]
                    if ((e1_start, e1_end), (e2_start, e2_end)) not in finalized_record["relations"].keys():
                        finalized_record["relations"][((e1_start, e1_end), (e2_start, e2_end))] = []
                    finalized_record["relations"][((e1_start, e1_end), (e2_start, e2_end))].append((e1_span_id,
                                                                                                e2_span_id,
                                                                                                relation["label"]))
            finalized_records.append(finalized_record)
        print("Max span count: ", max_span_count, ". Average span count: ",
              int(total_spans/len(documents)))
        return finalized_records

    def normalize_text(self, text: str) -> str:
        accents_translation_table = str.maketrans(
            "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸ",
            "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUY"
        )
        return text.translate(accents_translation_table)

    def locate_entity(self, entity_tokens: list, word_sequence: list) -> list:
        def locate_occurrences(target_token, token_sequence, is_end=False):
            positions = []
            for i, token in enumerate(token_sequence):
                if is_end:
                    if token == target_token[-1] or token.lower() == target_token[-1].lower():
                        positions.append(i)
                else:
                    if token == target_token[0] or token.lower() == target_token[0].lower():
                        positions.append(i)
            return positions

        found_range = (0, 0)
        start_positions = locate_occurrences(entity_tokens, word_sequence, is_end=False)
        end_positions = locate_occurrences(entity_tokens, word_sequence, is_end=True)
        if len(start_positions) == 1 and len(end_positions) == 1:
            return start_positions[0], end_positions[0]
        else:
            for start in start_positions:
                for end in end_positions:
                    if start <= end:
                        if word_sequence[start:end + 1] == entity_tokens:
                            found_range = (start, end)
                            break
            return found_range[0], found_range[1]



if __name__ == '__main__':
    dataset_config = {
        "step": "1",
        "pretrained_model": 'BERTS/bert_base_cased',
        "task": "_2022t12",
        "is_chinese": 0,
        "max_span_length": 7,
        "rel_span": 7,
        "star": 0,
        "save_path": "./processed_data/",
        "log_path": "./processed_data/",
        "negative_probability": 0,
        "sorted_entity": 1,
        "sorted_relation": 0,
        "duplicate_questions": 3,
        "token_type_ids": 1,
    }

    logging.basicConfig(filename=os.path.join(dataset_config["log_path"], "info.log"),
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger('root')
    logger.addHandler(logging.FileHandler(os.path.join(dataset_config["log_path"], "data.log"), 'w'))
    logger.info(sys.argv)
    logger.info(dataset_config)
    if dataset_config["task"] == "scierc":
        if dataset_config["step"] == "1":
            train_json_path = "./data/scierc/train.json"
            test_json_path = "./data/scierc/test.json"
            dev_json_path = "./data/scierc/dev.json"
            print("-" * 10, "Training", "-" * 10)
            storage_path = dataset_config["save_path"]
            train_category = "train"
            train_dataset = ProcessedDataset(dataset_config, train_json_path, train_category, storage_path, processing_stage="1")

            print("-" * 10, "Testing", "-" * 10)
            test_category = "test"
            test_dataset = ProcessedDataset(dataset_config, test_json_path, test_category, storage_path, processing_stage="1")

            print("-" * 10, "Validation", "-" * 10)
            dev_category = "dev"
            dev_dataset = ProcessedDataset(dataset_config, dev_json_path, dev_category, storage_path, processing_stage="1")
        if dataset_config["step"] == "2":
            storage_path = "../pred_result/pred_data/"
            test_json_path = "../pred_result/generate_result/pred_json_for_BF.json"
            test_category = "test"
            test_dataset = ProcessedDataset(dataset_config, test_json_path, test_category, storage_path, processing_stage="2")
    elif dataset_config["task"] == "_2017t10":
        if dataset_config["step"] == "1":
            train_json_path = "./data/_2017t10/train.json"
            test_json_path = "./data/_2017t10/test.json"
            dev_json_path = "./data/_2017t10/dev.json"
            print("-" * 10, "Training", "-" * 10)
            storage_path = dataset_config["save_path"]
            train_category = "train"
            train_dataset = ProcessedDataset(dataset_config, train_json_path, train_category, storage_path, processing_stage="1")

            print("-" * 10, "Testing", "-" * 10)
            test_category = "test"
            test_dataset = ProcessedDataset(dataset_config, test_json_path, test_category, storage_path, processing_stage="1")

            print("-" * 10, "Validation", "-" * 10)
            dev_category = "dev"
            dev_dataset = ProcessedDataset(dataset_config, dev_json_path, dev_category, storage_path, processing_stage="1")
    else:
        if dataset_config["step"] == "1":
            train_json_path = "./data/"+dataset_config["task"]+"/train.json"
            test_json_path = "./data/"+dataset_config["task"]+"/test.json"
            dev_json_path = "./data/"+dataset_config["task"]+"/test.json"  # dev - test
            print("-" * 10, "Training", "-" * 10)
            storage_path = dataset_config["save_path"]
            train_category = "train"
            train_dataset = ProcessedDataset(dataset_config, train_json_path, train_category, storage_path, processing_stage="1")

            print("-" * 10, "Testing", "-" * 10)
            test_category = "test"
            test_dataset = ProcessedDataset(dataset_config, test_json_path, test_category, storage_path, processing_stage="1")

            print("-" * 10, "Validation", "-" * 10)
            dev_category = "dev"
            dev_dataset = ProcessedDataset(dataset_config, dev_json_path, dev_category, storage_path, processing_stage="1")