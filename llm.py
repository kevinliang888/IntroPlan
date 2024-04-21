import openai
import signal


# @markdown LLM API call
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def lm_instruct(prompt,
       max_tokens=256,
       temperature=0,
       logprobs=None,
       stop_seq=None,
       logit_bias={
           32: 100.0,  #A (with space at front)
           33: 100.0,  #B (with space at front)
           34: 100.0,  #C (with space at front)
           35: 100.0,  #D (with space at front)
           36: 100.0,  #E (with space at front)
       },
       timeout_seconds=20):
    max_attempts = 5
    for _ in range(max_attempts):
        try:
            with timeout(seconds=timeout_seconds):
                response = openai.Completion.create(
                    model='gpt-3.5-turbo-instruct',
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=logprobs,
                    logit_bias=logit_bias,
                    stop=list(stop_seq) if stop_seq is not None else None,
                )
            break
        except openai.error.APIError as e:
          #Handle API error here, e.g. retry or log
          print(f"OpenAI API returned an API Error: {e}")
          pass
        except openai.error.APIConnectionError as e:
          #Handle connection error here
          print(f"Failed to connect to OpenAI API: {e}")
          pass
        except openai.error.RateLimitError as e:
          #Handle rate limit error (we recommend using exponential backoff)
          print(f"OpenAI API request exceeded rate limit: {e}")
          pass
    return response, response["choices"][0]["text"].strip()

def lm(messages, logprobs=True, top_logprobs=5, timeout_seconds=300, max_tokens=256, model="gpt-4-1106-preview", temperature=0,
                    logit_bias={
                       32: 100.0,  #A 
                       33: 100.0,  #B 
                       34: 100.0,  #C 
                       35: 100.0,  #D 
                       36: 100.0,  #E 
                   }):
    max_attempts = 5
    response = None
    messages = [{"role": "user", "content": messages}]
    for _ in range(max_attempts):
        try:
            with timeout(seconds=timeout_seconds):
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    logit_bias=logit_bias,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            break
        except openai.error.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
        except openai.error.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
        except openai.error.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")

    if response:
        return response, response["choices"][0]["message"]["content"].strip()
    else:
        return None, None


def lm_chat(messages, logprobs=True, top_logprobs=5, timeout_seconds=100, max_tokens=256, model="gpt-3.5-turbo-1106", temperature=0,
                    logit_bias={
                       32: 100.0,  #A 
                       33: 100.0,  #B 
                       34: 100.0,  #C 
                       35: 100.0,  #D 
                       36: 100.0,  #E 
                   }):
    max_attempts = 5
    response = None

    for _ in range(max_attempts):
        try:
            with timeout(seconds=timeout_seconds):
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    logit_bias=logit_bias,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            break
        except openai.error.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
        except openai.error.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
        except openai.error.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")

    if response:
        return response, response["choices"][0]["message"]["content"].strip()
    else:
        return None, None