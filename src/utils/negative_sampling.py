import numpy as np
import random
import os
import logging
import pickle


class RandomNegativeSampler():
    """
    Randomly sample candidates from a list of candidates.
    Args:
        candidates: list of str containing the candidates
        num_candidates_samples: int containing the number of negative samples for each query.
    """
    def __init__(self, candidates, num_candidates_samples, seed=42):
        random.seed(seed)
        self.candidates = candidates
        self.num_candidates_samples = num_candidates_samples
        self.name = "RandomNS"
        
    def sample(self, query_str, relevant_docs):
        """
        Samples from a list of candidates randomly.
        
        If the samples match the relevant doc, 
        then removes it and re-samples.
        Args:
            query_str: the str of the query. Not used here.
            relevant_docs: list with the str of the relevant documents, to avoid sampling them as negative sample.
        Returns:
            A triplet containing the list of negative samples, 
            whether the method had retrieved the relevant doc and 
            if yes its rank in the list.
        """
        sampled_initial = random.sample(self.candidates, self.num_candidates_samples)
        was_relevant_sampled = False
        relevant_doc_rank = -1
        sampled = []
        for i, d in enumerate(sampled_initial):
            if d in relevant_docs:
                was_relevant_sampled = True
                relevant_doc_rank = i
            else:
                sampled.append(d)

        while len(sampled) != self.num_candidates_samples:
            sampled = [d for d in random.sample(self.candidates, self.num_candidates_samples) if d not in relevant_docs]
        return sampled, was_relevant_sampled, relevant_doc_rank
