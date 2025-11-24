from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from jsonargparse import CLI

from needlehaystack import LLMNeedleHaystackTester, LLMMultiNeedleHaystackTester
from needlehaystack.evaluators import Evaluator, LangSmithEvaluator, OpenAIEvaluator
from needlehaystack.providers import Anthropic, ModelProvider, OpenAI, Cohere

load_dotenv()

@dataclass
class CommandArgs():
    """Command line arguments for LLM Needle in Haystack testing.
    
    LLMNeedleHaystackTester parameters:
    - model_to_test: The model to run the needle in a haystack test on. Default is None.
    - evaluator: An evaluator to evaluate the model's response. Default is None.
    - needle: The statement or fact which will be placed in your context ('haystack')
    - haystack_dir: The directory which contains the text files to load as background context. Only text files are supported
    - retrieval_question: The question with which to retrieve your needle in the background context
    - results_version: You may want to run your test multiple times for the same combination of length/depth, change the version number if so
    - num_concurrent_requests: Default: 1. Set higher if you'd like to run more requests in parallel. Keep in mind rate limits.
    - save_results: Whether or not you'd like to save your results to file. They will be temporarily saved in the object regardless. True/False. If save_results = True, then this script will populate a result/ directory with evaluation information. Due to potential concurrent requests each new test will be saved as a few file.
    - save_contexts: Whether or not you'd like to save your contexts to file. Warning these will get very long. True/False
    - final_context_length_buffer: The amount of context to take off each input to account for system messages and output tokens. This can be more intelligent but using a static value for now. Default 200 tokens.
    - context_lengths_min: The starting point of your context lengths list to iterate
    - context_lengths_max: The ending point of your context lengths list to iterate
    - context_lengths_num_intervals: The number of intervals between your min/max to iterate through
    - context_lengths: A custom set of context lengths. This will override the values set for context_lengths_min, max, and intervals if set
    - document_depth_percent_min: The starting point of your document depths. Should be int > 0
    - document_depth_percent_max: The ending point of your document depths. Should be int < 100
    - document_depth_percent_intervals: The number of iterations to do between your min/max points
    - document_depth_percents: A custom set of document depths lengths. This will override the values set for document_depth_percent_min, max, and intervals if set
    - document_depth_percent_interval_type: Determines the distribution of depths to iterate over. 'linear' or 'sigmoid'
    - seconds_to_sleep_between_completions: Default: None, set # of seconds if you'd like to slow down your requests
    - print_ongoing_status: Default: True, whether or not to print the status of test as they complete
    
    LLMMultiNeedleHaystackTester parameters:
    - multi_needle: True or False, whether to run multi-needle
    - needles: List of needles to insert in the context
    
    Other Parameters:
    - model_name: The name of the model you'd like to use. Should match the exact value which needs to be passed to the api. Ex: For OpenAI inference and evaluator models it would be gpt-4.1-mini.
    """
    
    provider: str = "openai"  # The provider of the model, available options are openai, anthropic, and cohere
    evaluator: str = "openai"  # The evaluator, which can either be a model or LangSmith
    model_name: str = "gpt-4.1-mini"  # Model name of the language model accessible by the provider
    evaluator_model_name: Optional[str] = "gpt-4.1-mini"  # Model name of the language model accessible by the evaluator
    needle: Optional[str] = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"  # The statement or fact which will be placed in your context ('haystack')
    haystack_dir: Optional[str] = "PaulGrahamEssays"  # The directory which contains the text files to load as background context
    retrieval_question: Optional[str] = "What is the best thing to do in San Francisco?"  # The question with which to retrieve your needle in the background context
    results_version: Optional[int] = 1  # You may want to run your test multiple times for the same combination of length/depth, change the version number if so
    context_lengths_min: Optional[int] = 1000  # The starting point of your context lengths list to iterate
    context_lengths_max: Optional[int] = 16000  # The ending point of your context lengths list to iterate
    context_lengths_num_intervals: Optional[int] = 35  # The number of intervals between your min/max to iterate through
    context_lengths: Optional[list[int]] = None  # A custom set of context lengths. This will override the values set for context_lengths_min, max, and intervals if set
    document_depth_percent_min: Optional[int] = 0  # The starting point of your document depths. Should be int > 0
    document_depth_percent_max: Optional[int] = 100  # The ending point of your document depths. Should be int < 100
    document_depth_percent_intervals: Optional[int] = 35  # The number of iterations to do between your min/max points
    document_depth_percents: Optional[list[int]] = None  # A custom set of document depths lengths. This will override the values set for document_depth_percent_min, max, and intervals if set
    document_depth_percent_interval_type: Optional[str] = "linear"  # Determines the distribution of depths to iterate over. 'linear' or 'sigmoid'
    num_concurrent_requests: Optional[int] = 1  # Default: 1. Set higher if you'd like to run more requests in parallel. Keep in mind rate limits.
    save_results: Optional[bool] = True  # Whether or not you'd like to save your results to file
    save_contexts: Optional[bool] = True  # Whether or not you'd like to save your contexts to file. Warning these will get very long
    final_context_length_buffer: Optional[int] = 200  # The amount of context to take off each input to account for system messages and output tokens
    seconds_to_sleep_between_completions: Optional[float] = None  # Default: None, set # of seconds if you'd like to slow down your requests
    print_ongoing_status: Optional[bool] = True  # Default: True, whether or not to print the status of test as they complete
    eval_set: Optional[str] = "multi-needle-eval-pizza-3"  # LangSmith evaluation set name
    multi_needle: Optional[bool] = False  # True or False, whether to run multi-needle
    needles: list[str] = field(default_factory=lambda: [  # List of needles to insert in the context
        " Figs are one of the secret ingredients needed to build the perfect pizza. ", 
        " Prosciutto is one of the secret ingredients needed to build the perfect pizza. ", 
        " Goat cheese is one of the secret ingredients needed to build the perfect pizza. "
    ])

def get_model_to_test(args: CommandArgs) -> ModelProvider:
    """
    Determines and returns the appropriate model provider based on the provided command arguments.
    
    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.
        
    Returns:
        ModelProvider: An instance of the specified model provider class.
    
    Raises:
        ValueError: If the specified provider is not supported.
    """
    match args.provider.lower():
        case "openai":
            return OpenAI(model_name=args.model_name)
        case "anthropic":
            return Anthropic(model_name=args.model_name)
        case "cohere":
            return Cohere(model_name=args.model_name)
        case _:
            raise ValueError(f"Invalid provider: {args.provider}")

def get_evaluator(args: CommandArgs) -> Evaluator:
    """
    Selects and returns the appropriate evaluator based on the provided command arguments.
    
    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.
        
    Returns:
        Evaluator: An instance of the specified evaluator class.
        
    Raises:
        ValueError: If the specified evaluator is not supported.
    """
    match args.evaluator.lower():
        case "openai":
            return OpenAIEvaluator(model_name=args.evaluator_model_name,
                                   question_asked=args.retrieval_question,
                                   true_answer=args.needle)
        case "langsmith":
            return LangSmithEvaluator()
        case _:
            raise ValueError(f"Invalid evaluator: {args.evaluator}")

def main():
    """
    The main function to execute the testing process based on command line arguments.
    
    It parses the command line arguments, selects the appropriate model provider and evaluator,
    and initiates the testing process either for single-needle or multi-needle scenarios.
    """
    args = CLI(CommandArgs, as_positional=False)
    args.model_to_test = get_model_to_test(args)
    args.evaluator = get_evaluator(args)
    
    if args.multi_needle == True:
        print("Testing multi-needle")
        tester = LLMMultiNeedleHaystackTester(**args.__dict__)
    else: 
        print("Testing single-needle")
        tester = LLMNeedleHaystackTester(**args.__dict__)
    tester.start_test()

if __name__ == "__main__":
    main()
