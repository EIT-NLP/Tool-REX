# evaluate embedding models

from tool_de.eval import eval_retrieval
from tool_de.config import _MODEL, _TASK
import os

model = _MODEL[0]
print(model)

task = ['all']
output_file = ','.join(task)+'.json'
results = eval_retrieval(model_name=model,
                        tasks=task,
                        category='all',
                        output_file=output_file,
                        is_inst=True)
print(results)
