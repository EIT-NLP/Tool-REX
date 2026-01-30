import argparse
import mteb
from mteb import MTEB
import logging
import os
import json
import multiprocessing 

from functools import partial
import logging
import math
from typing import Any, Callable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams

# Multiprocessing configuration moved to the entry script.
# Avoid duplicating it here to prevent conflicts.

from mteb.encoder_interface import Encoder
from mteb.evaluation.evaluators.RetrievalEvaluator import DenseRetrievalExactSearch
from mteb.model_meta import ModelMeta
from mteb.models.rerankers_custom import RerankerWrapper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolRank(RerankerWrapper):
    name: str = "ToolRank"

    def __init__(
        self,
        model_name_or_path: str = "Lux1997/Tool-Rank-4B",
        batch_size: int = 999999999999,
        context_size: int = 32000,
        max_output_tokens: int = 8192,
        fp_options: str = "float16",
        num_gpus: int = 1,
        device: str = "cuda",
        force_rethink: int = 0,
        dataset_prompt: str = None,
        **kwargs,
    ):
        """
        rank1 is a reasoning reranker model (using test-time compute) which generates a reasoning chain before deciding true or false
        Now modified to use Qwen pointwise prompt format

        Args:
            model_name_or_path: Path to the model or name of the model on HuggingFace Hub
            batch_size: Maximum batch size for processing (default: very large number to let vLLM handle batching)
            context_size: Maximum context length for the model (default: 4096)
            max_output_tokens: Maximum number of tokens to generate (default: 1024)
            fp_options: Floating point precision to use, e.g. 'float16' (default: 'float16')
            num_gpus: Number of GPUs to use for tensor parallelism (default: 1)
            device: Device to load the model on (default: 'cuda')
            force_rethink: Number of times to force model to rethink its answer (default: 0)
            **kwargs: Additional keyword arguments passed to parent RerankerWrapper
        """
        super().__init__(model_name_or_path, batch_size=batch_size, fp_options=fp_options, **kwargs)
        
        self.context_size = context_size
        self.max_output_tokens = max_output_tokens
        self.num_gpus = num_gpus
        self.device = device
        self.force_rethink = force_rethink
        self.model_name_or_path = model_name_or_path
        self.dataset_prompt = dataset_prompt

        # Initialize tokenizer with max length of 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache commonly used token IDs for Qwen format
        # Note: Qwen3-Reranker uses "yes"/"no" instead of "true"/"false"
        self.true_token = self.tokenizer("true", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("false", add_special_tokens=False).input_ids[0]
        
        # Debug: Print token IDs to ensure they are correct
        print(f"True token ID: {self.true_token}, False token ID: {self.false_token}")
        print(f"True token text: '{self.tokenizer.decode([self.true_token])}'")
        print(f"False token text: '{self.tokenizer.decode([self.false_token])}')")
        
        # Qwen-specific prefix and suffix tokens
        # Update system prompt to match "yes"/"no" tokens
        self.prefix = "<|im_start|>system\nJudge whether the Tool Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"true\" or \"false\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        # self.suffix = "<|im_end|>\n<|im_start|>assistant\n"
        
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        # Lazy initialize the model to avoid CUDA reinitialization issues
        self.model = None
        self._model_initialized = False
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,  # Only need to generate one token (true/false)
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],  # Restrict to only true/false tokens
            skip_special_tokens=False
        )

    def _initialize_model(self):
        """Lazy initialize the model to avoid CUDA reinitialization issues."""
        if self._model_initialized:
            return
        logger.info("Initializing vLLM model...")

        # 1) Only set multiprocessing in the entry script; do not set it here

        # 2) Compute available GPUs -> tensor_parallel_size
        import torch
        gpu_count = torch.cuda.device_count()
        tensor_parallel_size = min(max(gpu_count, 1), self.num_gpus if self.num_gpus else 1)
        logger.info(f"Detected {gpu_count} GPU(s), tensor_parallel_size = {tensor_parallel_size}")

        # 3) Safely determine supported max length
        try:
            # tokenizer is already loaded in __init__
            model_max = getattr(self.tokenizer, "model_max_length", None)
            if not model_max or model_max == int(1e30):  # Some tokenizers return a huge value
                model_max = 8192  # Conservative default
        except Exception:
            model_max = 8192
        max_model_len = min(int(self.context_size or 8192), int(model_max))
        logger.info(f"max_model_len = {max_model_len} (tokenizer max {model_max})")

        # 4) Initialize vLLM (remove incompatible params; dtype=auto)
        try:
            from vllm import LLM
            self.model = LLM(
                model=self.model_name_or_path,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
                max_model_len=max_model_len,
                dtype="auto",                # or torch.float16
                gpu_memory_utilization=0.85, # Slightly higher throughput
                enforce_eager=False          # Use default False unless debugging graph
            )
            self._model_initialized = True
            logger.info("vLLM model initialized")
        except TypeError as te:
            logger.error(f"vLLM init parameters incompatible (TypeError): {te}")
            logger.info("Falling back to transformers path")
            self._initialize_transformers_model()
            self._model_initialized = True
        except Exception as e:
            logger.error(f"vLLM initialization failed: {e}")
            logger.info("Falling back to transformers path")
            self._initialize_transformers_model()
            self._model_initialized = True

    # def _initialize_model(self):
    #     """Lazy initialize the model to avoid CUDA reinitialization issues."""
    #     if not self._model_initialized:
    #         logger.info("Initializing vLLM model...")
    #         try:
    #             # Detect available GPU count
    #             import torch
    #             gpu_count = torch.cuda.device_count()
    #             logger.info(f"Detected {gpu_count} GPU(s)")
                
    #             # Set tensor_parallel_size based on GPU count
    #             if gpu_count >= 2:
    #                 tensor_parallel_size = 2
    #                 logger.info("Using 2-GPU parallelism")
    #             else:
    #                 tensor_parallel_size = 1
    #                 logger.info("Using single GPU")
                
    #             self.model = LLM(
    #                 model=self.model_name_or_path,
    #                 tensor_parallel_size=tensor_parallel_size,
    #                 trust_remote_code=True,
            #         max_model_len=self.context_size,
            #         gpu_memory_utilization=0.7,
            #         dtype=self.fp_options,
            #         disable_log_stats=True,
            #         enforce_eager=True,
            #     )
            #     self._model_initialized = True
            #     logger.info("vLLM model initialized")
            # except Exception as e:
            #     logger.error(f"vLLM initialization failed: {e}")
            #     # If vLLM fails, try transformers
            #     self._initialize_transformers_model()
            #     self._model_initialized = True

    def _initialize_transformers_model(self):
        """Fallback to transformers initialization."""
        logger.info("Initializing model with transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        logger.info("Transformers model initialized")

    def truncate_input(self, text: str) -> str:
        """
        Truncate the input text to the context size, ensuring it ends with the correct suffix
        """
        tokens = self.tokenizer(text)["input_ids"]
        if len(tokens) >= self.context_size:
            # Calculate how much space we need for the suffix
            suffix_length = len(self.suffix_tokens)
            max_content_length = self.context_size - suffix_length - 50  # Leave some buffer
            
            # Split the text to find the document part that can be truncated
            if "<Document>:" in text and "<|im_end|>" in text:
                # Find the document section
                doc_start = text.find("<Document>:") + len("<Document>:")
                doc_end = text.find("<|im_end|>", doc_start)
                
                prefix_part = text[:doc_start]
                document_part = text[doc_start:doc_end]
                suffix_part = text[doc_end:]
                
                # Calculate how many tokens we have for the document
                prefix_tokens = len(self.tokenizer(prefix_part)["input_ids"])
                suffix_tokens = len(self.tokenizer(suffix_part)["input_ids"])
                available_doc_tokens = max_content_length - prefix_tokens - suffix_tokens
                
                if available_doc_tokens > 0:
                    # Truncate the document part
                    doc_tokens = self.tokenizer(document_part)["input_ids"]
                    if len(doc_tokens) > available_doc_tokens:
                        truncated_doc_tokens = doc_tokens[:available_doc_tokens]
                        truncated_document = self.tokenizer.decode(truncated_doc_tokens, skip_special_tokens=True)
                        truncated_text = prefix_part + truncated_document + suffix_part
                    else:
                        truncated_text = text
                else:
                    # Fallback: simple truncation
                    truncated_tokens = tokens[:max_content_length]
                    truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    # Ensure proper ending
                    if not truncated_text.endswith("</think>\n\n"):
                        truncated_text = truncated_text.rstrip() + "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            else:
                # Fallback: simple truncation
                truncated_tokens = tokens[:max_content_length]
                truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
            final_length = len(self.tokenizer(truncated_text)["input_ids"])
            print(f"Truncated prompt from {len(tokens)} to {final_length} tokens")
            return truncated_text
        else:
            return text

    def _process_with_vllm(self, prompts):
        """
        vLLM is significantly faster than HF, so we use it by default. 
        Since prompt ends with </think>, model should directly output true/false as first token.

        Args:
            prompts: The prompts to generate from

        Returns:
            outputs: The outputs from the vLLM model
        """
        # Ensure model is initialized
        self._initialize_model()
        
        # Check whether we are using transformers (vLLM model has no generate method)
        if not hasattr(self.model, 'generate'):
            return self._process_with_transformers(prompts)
        
        # Truncate prompts that are too long
        prompts = [self.truncate_input(prompt) for prompt in prompts]
        outputs = self.model.generate(prompts, self.sampling_params)

        all_outputs = []
        all_output_token_counts = []
        all_scores = []
        
        for i, output in enumerate(outputs):
            text = output.outputs[0].text.strip()
            token_count = len(output.outputs[0].token_ids)
            
            try:
                # Get logits for the last generated token (following Qwen3-Reranker example)
                # Note: use [-1] to access logprobs of the last token, not [0]
                final_token_logits = output.outputs[0].logprobs[-1]
                
                # Check if yes/no tokens exist; use defaults if missing
                if self.true_token not in final_token_logits:
                    true_logit = -10  # Very small probability
                else:
                    true_logit = final_token_logits[self.true_token].logprob
                    
                if self.false_token not in final_token_logits:
                    false_logit = -10  # Very small probability
                else:
                    false_logit = final_token_logits[self.false_token].logprob
                
                true_score = math.exp(true_logit)
                false_score = math.exp(false_logit)
                score = true_score / (true_score + false_score)
                    
            except Exception as e:
                print(f"Error processing output {i}: {e}")
                print(f"Output: {output.outputs[0]}")
                print(f"Generated text: '{text}'")
                if hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs:
                    print(f"Available logprobs: {list(output.outputs[0].logprobs[-1].keys())}")
                # Fallback: use a default score
                score = 0.5
            
            all_outputs.append(text)
            all_output_token_counts.append(token_count)
            all_scores.append(score)

        return all_outputs, all_output_token_counts, all_scores

    def format_instruction(self, instruction, query, document):
        """Format instruction using rank1's original format within Qwen template"""
        return "Query: {query}\nPassage: {document}".format(
            query=query, document=document
        )

    def return_prompt(self, query, doc_content, prompt) -> str:
        """Create prompt using Qwen's format - following original rank1.py design"""
        if prompt:
            final_query = prompt.replace("FILL_QUERY_HERE", query)
        else:
            final_query = query
        
        formatted_content = self.format_instruction(None, final_query, doc_content)
        
        # Construct full prompt with Qwen's format
        full_prompt = self.prefix + formatted_content + self.suffix
        return full_prompt

    def _prepare_prompts_for_rethink(self, prompts: List[str], texts: List[str], rethink_text: str = "Wait") -> List[str]:
        """Prepare prompts for the rethinking step."""
        full_texts = [p + t for p, t in zip(prompts, texts)]
        stripped_texts = [t.split("</think>")[0] for t in full_texts]
        just_generated_texts = [t.split("</think>")[0] for t in full_texts]
        return [s + f"\n{rethink_text}" for s in stripped_texts], just_generated_texts

    @torch.inference_mode()
    def predict(self, input_to_rerank, **kwargs):
        """This is setup to run with mteb but can be adapted to your purpose"""
        inputs = list(zip(*input_to_rerank))
        if len(input_to_rerank[0]) == 2:
            queries, passages = inputs
            instructions = None
        else:
            queries, passages, instructions = inputs

        if instructions is not None and instructions[0] is not None:
            queries = [f"{q} {i}".strip() if q.strip() != i.strip() else q.strip() for i, q in zip(instructions, queries)]

        if isinstance(passages[0], dict):
            passages = [f"{v['title']} {v['text']}" if 'title' in v else v['text'] for v in passages]

        prompts = [
            self.return_prompt(query, passage, self.dataset_prompt)
            for query, passage in zip(queries, passages)
        ]
        # print(f"Example prompt: ```\n{prompts[0]}\n```")

        texts, token_counts, scores = self._process_with_vllm(prompts)
        while self.force_rethink:
            revised_prompts, previously_generated_texts = self._prepare_prompts_for_rethink(prompts, texts)
            new_texts, new_token_counts, new_scores = self._process_with_vllm(revised_prompts)
            # add to the previous output
            texts = [prev + f"\n{rethink_text}" + f"{new_text}" for prev, new_text in zip(texts, new_texts)]
            scores = new_scores
            token_counts = [prev_token_count + new_token_count for prev_token_count, new_token_count in zip(token_counts, new_token_counts)]
            self.force_rethink -= 1

        return scores

    def compute_rank_score(self, query: str, tools: List[str], instruction: str = None):
        """
        Adapter for the RankModel interface in eval.py.
        Args:
            query: Query text.
            tools: List of tool documents.
            instruction: Optional instruction text.
        Returns:
            scores: Score list for each tool.
        """
        # Build input pairs: [(query, tool1), (query, tool2), ...]
        input_pairs = []
        for tool in tools:
            if instruction:
                # Combine instruction with query when provided
                combined_query = f"{instruction} {query}".strip()
            else:
                combined_query = query
            input_pairs.append((combined_query, tool))
        
        # Call predict to get scores
        scores = self.predict(input_pairs)
        return scores 
