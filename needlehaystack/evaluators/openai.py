import os

from .evaluator import Evaluator
from openai import OpenAI

class OpenAIEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""}

    def __init__(self,
                 model_name: str = "gpt-4.1-mini",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 true_answer: str = None,
                 question_asked: str = None,):
        """
        :param model_name: The name of the model.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """

        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked

        api_key = os.getenv('NIAH_EVALUATOR_API_KEY')
        if (not api_key):
            raise ValueError("NIAH_EVALUATOR_API_KEY must be in env for using openai evaluator.")

        self.api_key = api_key
        base_url = os.getenv('BASE_URL')
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )

    def evaluate_response(self, response: str) -> int:
        prompt = f"""
        Compare the following response to the reference answer and score it based on accuracy:
        
        Question: {self.question_asked}
        Reference Answer: {self.true_answer}
        Response to Evaluate: {response}
        
        {self.CRITERIA['accuracy']}
        
        Provide only a numerical score (1, 3, 5, 7, or 10):
        """
        
        try:
            eval_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            
            score_text = eval_response.choices[0].message.content.strip()
            return int(score_text)
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return 1
