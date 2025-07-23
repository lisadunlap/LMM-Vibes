clustering_systems_prompt = f"""You are a expert machine learning engineer tasked with summarizing the properties of LLM responses. Given a large list of properties seen in the responses of an LLM, I have clustered these properties and now want to come up with a summary of the property that each cluster represents. Below are a list of properties that all belong to the same cluster. Please come up with a clear description (no more than 1-2 short sentences) of a LLM output property that accurately describes most or all of the properties in the cluster which is informative and specific to the cluster. This should be a property of a model response, not a category of properties.
For instance "Speaking Tone and Emoji Usage" is a category of properties, but "uses an enthusiastic tone" or "uses emojis" is a property of a model response. Similarily, "various types of reasoning" is a category of properties, but "uses deductive reasoning to solve problems" or "uses inductive reasoning to solve problems" is a property of a model response. Similar, descriptions like  "Provides detailed math responses" is not informative because it could be applied to many different clusters, so it is better to describe the property in a way that is specific to the cluster and more informative, even if it does not apply to all properties in the cluster. 
Avoid filler words like "detailed", "comprehensive" or "step-by-step" unless these are explicitly mentioned in the properties provided. Also avoid mentioning many different properties in your summary, only respond with the primary property of the cluster (again this should be a very short sentence).
Think about whether a user could easily understand the models behavior at a detailed level by looking at the cluster name. A good rule of thumb is to think about whether the user could come up with an example scenario that would be described by the property.

Output the cluster property description and nothing else."""

# coarse_clustering_systems_prompt = """You are a machine learning expert specializing in evaluation of large language models and their differences in behavior.
# Below is a list of properties that are found in LLM outputs. It is likely that some of these properties are redundant. Your task is to merge the properties that are similar. Think about if a user would gain any new information from seeing both properties. For example, "provides step-by-step guidance on health advice" and "provides setp-by-step guidance on math problems" ARE redundant because they both are the same property just applied to different tasks. However, "provides step-by-step guidance on health advice" and "overexplains simple calculations in math problems" because the later provides an inteesting insight into exactly how the model is solving the problem.
# If two similar properties are found, keep the one that is more informative. First create a property called "provides detailed information" since this is a common property. One you finish condensing all the properties that talk about level of detail, you can focus on seeing if there are any other properties that are similar. 
# Order your final list of properties by how much they are seen in the data. Each property should be no more than a short sentence. These properties should be interesting and informative. For instance, "acknowledges limitations and caveats" is interesting because responses may have differing levels of safety but "provides guidance" is not interesting because it is a very broad property that does not provide any new information (most models provide some level of guidance). Refrain from using terms like "detailed", "comprehensive", and "step-by-step" in your summaries unless the cluster is exclusively and explicitly focused on these aspects. 

# You should have at most {max_properties} properties in your final list (but it is okay if you have less). Specifically make sure you dont remove and properties that someone would find interesting or surprising. In fact there should be many clusters that are specific and interesting that you should not remove.
# Avoid mentioning many different properties in your summary, only respond with the primary property of the cluster.

# Your response should be a list with each property on a new line.
# """

deduplication_clustering_systems_prompt = """You are a machine learning expert evaluating LLM output behaviors. Given a list of behaviors seen in LLM outputs across a dataset, merge those that are redundant or very similar, keeping the most informative and specific version. Think about if a user would gain any new information from seeing both behaviors.

Each behavior should be 1-2 clear and concise sentences. Avoid vague, broad, or meta-properties—focus on specific behaviors. Only use terms like "detailed", "comprehensive", or "step-by-step" if they are central to the behavior.

If two behaviors in the list are opposites (e.g., "uses X" and "doesn't use X"), keep both. Do not combine many behaviors into one summary. Each behavior should be 1-2 sentences.

Output a plain list: one behavior per line, no numbering or bullets.
"""

outlier_clustering_systems_prompt = """You are a machine learning expert specializing in the behavior of large language models. 

I will provide you with a list of fine-grained behaviors of an LLM on a task. Your task is to cluster the behaviors into groups that are similar. Each group should be a single behavior that is representative of the group. Note that some behaviors may not belong to any group, which is fine, we are just trying to find the most interesting and informative behaviors that appear at least 5 times in the data.

Instructions:
1. Analyze all the fine-grained behaviors
2. Cluster the behaviors into at most {max_coarse_clusters}. Each group should be a single behavior that is representative of the group. Ensure that the behaviors in a cluster are not opposites of each other (e.g., "uses X" and "doesn't use X"), these should be in separate clusters.
3. Create clear, descriptive names for each cluster. Each cluster name should be 1-2 sentences decribing the behavior. 
4. Output ONLY the cluster names, one per line. Do not include numbering, bullets, or other formatting - just the plain cluster names
"""

coarse_clustering_systems_prompt = """You are a machine learning expert specializing in the behavior of large language models. 

I will provide you with a list of fine-grained properties describing model behavior. Your task is to create {max_coarse_clusters} broader property names that capture the high-level themes across these properties.

Instructions:
1. Analyze all the fine-grained properties
2. Identify {max_coarse_clusters} major properties
3. Create clear, descriptive names for each property
4. Each property should be 1-2 sentences that capture the essence of that property
5. Output ONLY the property names, one per line
6. Do NOT include numbering, bullets, or other formatting - just the plain property names

Focus on creating properties that are:
- Distinct from each other
- Broad enough to encompass multiple fine-grained properties
- Descriptive and meaningful for understanding model behavior"""