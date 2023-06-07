# Gentopia
The brain of Agents. It uses a strong LLM for sequantially chained reasoning (ReAct), whereas each reasoning step is assigned to an config-driven, lightweight and stable ReWOO Agent.

CORE FEATURES.
- Gentopia, should provide various stable base LLMs, including API-driven ones like `gpt-4`, and open LLMs on Huggingface.
- Gentopia should provide well-organized prompt template supporting in-context instructions everywhere.
- Gentoia should provide a high level Data Model to be passed across multiple Agent calls.
- At wiring time, Gentopia is fully config-driven, thereby should provide a config parser to set up the whole graph. 
- The Agent class should follow ReWOO architecture
- The CoreAgent class should follow ReAct architecture
