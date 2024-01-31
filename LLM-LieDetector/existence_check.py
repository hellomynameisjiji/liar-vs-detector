
import dotenv, os
import openai
import pickle

from lllm.questions_loaders import Sciq

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


## TODO: Load a LLM model
dataset = Sciq()


## TODO: Compute the accuracy (logprobs)


## TODO: Load a pretrained detector
DETECTOR_PATH = 'results/trained_classifiers/logistic_binary_classifier_all_probes.pkl'

with open(DETECTOR_PATH, 'rb') as f:
    classifier_all = pickle.load(f)

## TODO: Compute the prediction (48 elicitation questions - 48-d probs)
    
    

